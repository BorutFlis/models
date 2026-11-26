"""Configuration and cohort eligibility for the one-year experiment."""

import json
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parents[1]
CONFIG = json.loads((SCRIPT_DIR / "config.json").read_text(encoding="utf-8"))
IDENTITY = ["patid", "date"]
OUTCOMES = ["death_event", "days_to_event"]
TARGET = "death_within_1_year"
SCORE = "focal_oof_score"


def filter_cohort(data, hospitalization_columns):
    """Apply the two eligibility rules in order and record attrition."""
    if not hospitalization_columns:
        raise ValueError("No hospitalization columns found in the source schema.")
    if data.patid.isna().any() or data.patid.duplicated().any():
        raise ValueError("Expected one record per patient and complete patient IDs.")

    index_date = pd.to_datetime(data.date, errors="coerce")
    registration = pd.to_datetime(data.regstartdate, errors="coerce")
    followup = (index_date - registration).dt.days
    registered = followup.gt(CONFIG["min_registration_days_exclusive"])
    hospitalized = data[hospitalization_columns].notna().any(axis=1)
    eligible = registered & hospitalized
    flow = pd.DataFrame({
        "stage": [
            "Patients in source dataset",
            "At least one calendar year since registration",
            "At least one populated hospitalization field",
        ],
        "included": [len(data), int(registered.sum()), int(eligible.sum())],
        "excluded": [0, int((~registered).sum()), int((registered & ~hospitalized).sum())],
    })
    return data.loc[eligible].copy(), flow
