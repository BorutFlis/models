"""Offline regression tests; no patient data or network access."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

try:
    from . import life_preserved as life
except ImportError:
    import life_preserved as life


def mapping():
    return pd.DataFrame([
        {"LIFE_Preserved_variable": label, "high_risk_HES_schema_attribute": source}
        for label, source in [("Age", "Phy_Age"), ("Sex", "Phy_Sex"),
                              ("Heart rate", None), ("Haemoglobin", "Blo_Hb__last")]
    ])


class LifeTests(unittest.TestCase):
    def test_interaction_retries_with_fresh_elements(self):
        try:
            from selenium.common.exceptions import ElementNotInteractableException, StaleElementReferenceException
        except ImportError:
            self.skipTest("Selenium is required for interaction tests")
        for failure in (ElementNotInteractableException, StaleElementReferenceException):
            driver = Mock()
            driver.find_elements.return_value = []
            first, second = Mock(), Mock()
            driver.find_element.side_effect = [first, second]
            action = Mock(side_effect=[failure("UI changing"), None])
            life.interact_when_ready(driver, ("id", "age"), action, "fill age", timeout=1)
            self.assertEqual(action.call_count, 2)
            self.assertIs(action.call_args.args[0], second)

    def test_hidden_element_is_not_used_and_timeout_names_field(self):
        try:
            from selenium.common.exceptions import TimeoutException
        except ImportError:
            self.skipTest("Selenium is required for interaction tests")
        driver, action = Mock(), Mock()
        driver.find_elements.return_value = []
        driver.find_element.return_value.is_displayed.return_value = False
        with self.assertRaisesRegex(TimeoutException, "fill numeric input age"):
            life.interact_when_ready(driver, ("id", "age"), action,
                                     "fill numeric input age", timeout=0.01)
        action.assert_not_called()

    def test_modal_blocks_input_interaction_until_closed(self):
        try:
            import selenium
        except ImportError:
            self.skipTest("Selenium is required for interaction tests")
        driver, overlay, element, action = Mock(), Mock(), Mock(), Mock()
        overlay.is_displayed.return_value = True
        driver.find_elements.side_effect = [[overlay], []]
        driver.find_element.return_value = element
        life.interact_when_ready(driver, ("id", "age"), action, "fill age", timeout=1)
        self.assertEqual(driver.find_element.call_count, 1)
        action.assert_called_once_with(element)

    def test_source_sex_is_reversed_but_site_sex_is_not(self):
        self.assertEqual(life.prepare_profile({"Phy_Sex": 0}, mapping())["inputs"]["sex"], "1")
        self.assertEqual(life.prepare_profile({"sex": 0}, mapping())["inputs"]["sex"], "0")
        self.assertEqual(life.prepare_profile({"Phy_Sex": "F"}, mapping())["inputs"]["sex"], "0")

    def test_missing_values_and_hb_units_are_audited(self):
        result = life.prepare_profile({"Phy_Age": float("nan")}, mapping(), hb_unit="mmolL")
        self.assertEqual(result["imputed"]["age"], 70)
        self.assertAlmostEqual(result["inputs"]["hb"], 14 / 1.6114)
        self.assertEqual(life.prepare_profile({"Blo_Hb__last": 8.5}, mapping(), hb_unit="mmolL")["inputs"]["hb"], 8.5)

    def test_invalid_values_fail_instead_of_silent_clipping(self):
        for row in ({"age": float("inf")}, {"heartrate": float("inf")}, {"heartrate": 130}, {"sex": "unknown"}):
            with self.assertRaises(ValueError):
                life.prepare_profile(row, mapping())

    def test_age_and_bmi_are_clipped_and_audited(self):
        for row, expected in [
            ({"age": 96.13150684931507, "bmi": 56.4}, {"age": 89, "bmi": 55}),
            ({"age": 30, "bmi": 16}, {"age": 40, "bmi": 18}),
        ]:
            profile = life.prepare_profile(row, mapping())
            for field, value in expected.items():
                self.assertEqual(profile["inputs"][field], value)
                self.assertEqual(profile["clipped"][field],
                                 {"original": row[field], "submitted": value})
            self.assertNotIn("age", profile["imputed"])
        self.assertEqual(life.prepare_profile({"age": 89, "bmi": 55}, mapping())["clipped"], {})

    def test_additional_ranges_are_clipped_with_hb_units(self):
        for row, expected in [
            ({"gfr": 100, "hb": 12, "ntprobnp": 25000}, {"gfr": 90, "hb": 11, "ntprobnp": 20000}),
            ({"gfr": 10, "hb": 2, "ntprobnp": -1}, {"gfr": 15, "hb": 3, "ntprobnp": 0}),
        ]:
            result = life.prepare_profile(row, mapping(), hb_unit="mmolL")
            for field, expected_value in expected.items():
                self.assertEqual(result["inputs"][field], expected_value)
                self.assertEqual(result["clipped"][field]["original"], row[field])
        for observed, expected in [(2, 3 * 1.6114), (20, 11 * 1.6114)]:
            result = life.prepare_profile({"hb": observed}, mapping(), hb_unit="gdL")
            self.assertAlmostEqual(result["inputs"]["hb"], expected)
        self.assertEqual(life.prepare_profile({"hb": 14}, mapping())["inputs"]["hb"], 14)

    def test_means_use_observed_values_only(self):
        reference = pd.DataFrame({"Phy_Age": [60, 80, None], "Phy_Sex": [0, 0, 1]})
        defaults = life.fit_reference_defaults(reference, mapping())
        self.assertEqual(defaults["age"], 70)
        self.assertEqual(defaults["sex"], "1")
        self.assertEqual(defaults["heartrate"], 70)

    def test_unmapped_rows_keep_empty_selectors(self):
        frame = pd.concat([mapping(), pd.DataFrame([{"LIFE_Preserved_variable": "Outcome"}])])
        result = life.attach_identifiers(frame)
        self.assertTrue(pd.isna(result.iloc[-1].shiny_input_selector))
        self.assertEqual(result.iloc[1].shiny_input_selector, 'input[name="sex"]')

    def test_parse_all_output_formats(self):
        for text, number, unit in [("21.3%", 21.3, "%"), ("84.7 years", 84.7, "years"), ("0years", 0, "years")]:
            result = life.parse_result(text, "Label")
            self.assertEqual((result["value"], result["unit"]), (number, unit))
        with self.assertRaises(ValueError):
            life.parse_result("Please enter risk profile", "Risk")

    def test_unavailable_results_are_json_safe_and_not_zero(self):
        for text, unit in [("NA%", "%"), ("NA years", "years"), ("NAyears", "years")]:
            result = life.parse_result(text, "Risk")
            self.assertIsNone(result["value"])
            self.assertEqual(result["unit"], unit)
            self.assertEqual(result["status"], "unavailable")
            self.assertEqual(result["text"], text)
            self.assertIn('"value": null', json.dumps(result, allow_nan=False))

    def test_retry_resume_and_changed_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            def success(profile, directory, **kwargs):
                directory.mkdir(parents=True, exist_ok=True)
                html = directory / "assessment.html"
                html.write_text("fixture")
                return {**profile, "plots": {}, "html": str(html), "metrics": {}}
            predict = Mock(side_effect=[RuntimeError("offline"), success(life.prepare_profile({}, mapping()), Path(temporary)/"fixture")])
            frame = pd.DataFrame({"Phy_Age": [70]}, index=["row-a"])
            result = life.run_batch(frame, mapping(), predict, temporary, delay=0)
            self.assertEqual(result.iloc[0].status, "ok")
            self.assertEqual(predict.call_count, 2)
            life.run_batch(frame, mapping(), predict, temporary, delay=0)
            self.assertEqual(predict.call_count, 2)
            changed = Mock(side_effect=success)
            frame.loc["row-a", "Phy_Age"] = 71
            life.run_batch(frame, mapping(), changed, temporary, delay=0)
            self.assertEqual(changed.call_count, 1)

    def test_invalid_row_does_not_stop_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame = pd.DataFrame({"Phy_Age": [float("inf"), 70]})
            predict = Mock(side_effect=RuntimeError("network"))
            result = life.run_batch(frame, mapping(), predict, temporary, retries=0, delay=0)
            self.assertEqual(list(result.status), ["error", "error"])
            self.assertEqual(predict.call_count, 1)

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            life.run_batch(pd.DataFrame(index=[1, 1]), mapping(), Mock(), "unused")


if __name__ == "__main__":
    unittest.main()
