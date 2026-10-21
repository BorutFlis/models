"""Tests for interval boundaries, censoring and cumulative probability rules."""
import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from early_diagnosis.scripts.conditional_risk.reproduce_experiments import (
    create_targets, cumulative_risks, load_cohort, patient_splits,
)


class ConditionalRiskTests(unittest.TestCase):
    def test_all_timepoints_and_reproducible_patient_sampling(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / "id_maps").mkdir()
            (directory / "id_maps/29.json").write_text(json.dumps(
                {"0": "patient_a", "1": "patient_b", "2": "patient_c"}))
            frame = pd.DataFrame({
                "ID": np.repeat([0, 1, 2], 3),
                "Timepoint": list(pd.date_range("2020-01-01", periods=3)) * 3,
                "days_to_HFD": [2, 1, 0] + [np.nan] * 6,
                "Dia_HFD_patient": [1] * 3 + [0] * 6,
                "Blo_NT": [10, np.nan, 30] + [np.nan] * 6,
                "Phy_Age": [60] * 9,
            })
            frame.to_csv(directory / "29.csv", index=False)
            cohort, counts = load_cohort(directory, ["29.csv"], 3, 42)
            self.assertEqual(len(cohort), 8)  # Diagnosis-day row excluded.
            self.assertEqual(counts.sampled_patients.iat[0], 3)
            self.assertEqual(cohort.groupby("ID").size().to_dict(),
                             {"patient_a": 2, "patient_b": 3, "patient_c": 3})
            self.assertEqual(cohort.loc[cohort.ID.eq("patient_a"), "Blo_NT"].tolist(), [10, 10])
            first, _ = load_cohort(directory, ["29.csv"], 2, 42)
            second, _ = load_cohort(directory, ["29.csv"], 2, 42)
            self.assertEqual(first.ID.nunique(), 2)
            pd.testing.assert_frame_equal(first, second)

    def test_patient_folds_keep_all_timepoints_together(self):
        cohort = pd.DataFrame({"ID": np.repeat(np.arange(12), 3),
            "days_to_HFD": np.repeat([100.0] * 6 + [np.nan] * 6, 3)})
        gather_test = []
        for train, test in patient_splits(cohort, 3, 42):
            self.assertFalse(set(cohort.iloc[train].ID) & set(cohort.iloc[test].ID))
            self.assertTrue(cohort.iloc[test].groupby("ID").size().eq(3).all())
            gather_test.extend(test)
        self.assertEqual(sorted(gather_test), list(range(len(cohort))))

    def test_interval_boundaries_and_censoring(self):
        cohort = pd.DataFrame({
            "days_to_HFD": [100, 365.25, 500, 730.5, 1000, 2000,
                            np.nan, np.nan, np.nan, 0, -1],
            "followup_days": [0, 0, 0, 0, 0, 0, 2000, 500, 100, 2000, 2000],
        })
        targets = create_targets(cohort).fillna(-9)
        self.assertEqual(targets.conditional_0_1Y.tolist(),
                         [1, 1, 0, 0, 0, 0, 0, 0, -9, -9, -9])
        self.assertEqual(targets.conditional_1_2Y.tolist(),
                         [-9, -9, 1, 1, 0, 0, 0, -9, -9, -9, -9])
        self.assertEqual(targets.conditional_2_5Y.tolist(),
                         [-9, -9, -9, -9, 1, 0, 0, -9, -9, -9, -9])
        self.assertEqual(targets.HF_5Y.tolist(),
                         [1, 1, 1, 1, 1, 0, 0, -9, -9, -9, -9])

    def test_monotonicity_for_extremes_and_arbitrary_hazards(self):
        hazards = np.vstack([[0, 0, 0], [1, 0, 0], [0, 0, 1],
                             [0.1, 0.15, 0.2],
                             np.random.default_rng(42).random((1000, 3))])
        risks = cumulative_risks(hazards)
        self.assertTrue((np.diff(risks, axis=1) >= 0).all())
        np.testing.assert_allclose(risks[3], [0.1, 0.235, 0.388])
        self.assertTrue(((risks >= 0) & (risks <= 1)).all())

    def test_invalid_hazards_fail(self):
        for hazards in ([[0, 1]], [[0, np.nan, 0]], [[0, -0.1, 0]], [[0, 1.1, 0]]):
            with self.assertRaises(ValueError):
                cumulative_risks(hazards)


if __name__ == "__main__":
    unittest.main()
