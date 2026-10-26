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
        for row in ({"age": 90}, {"heartrate": float("inf")}, {"sex": "unknown"}):
            with self.assertRaises(ValueError):
                life.prepare_profile(row, mapping())

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
            frame = pd.DataFrame({"Phy_Age": [95, 70]})
            predict = Mock(side_effect=RuntimeError("network"))
            result = life.run_batch(frame, mapping(), predict, temporary, retries=0, delay=0)
            self.assertEqual(list(result.status), ["error", "error"])
            self.assertEqual(predict.call_count, 1)

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            life.run_batch(pd.DataFrame(index=[1, 1]), mapping(), Mock(), "unused")


if __name__ == "__main__":
    unittest.main()
