"""Cohort construction and source resolution for the HES mortality experiment."""

import json
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parents[1]
CONFIG = json.loads((SCRIPT_DIR / "config.json").read_text(encoding="utf-8"))


def resolve_source(path):
    """Read a cached LFS object directly when the working tree contains a pointer."""
    with path.open("rb") as source:
        if source.readline().strip() != b"version https://git-lfs.github.com/spec/v1":
            return path
        oid = source.readline().decode().strip().removeprefix("oid sha256:")
        size = int(source.readline().decode().strip().removeprefix("size "))
    cached = PROJECT.parent / ".git" / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
    if not cached.is_file() or cached.stat().st_size != size:
        raise FileNotFoundError(f"Restore the Git LFS content for {path} before running.")
    return cached


def filter_cohort(data, hospitalization_columns):
    """Apply the three eligibility rules in order and count patients at each step."""
    if not hospitalization_columns:
        raise ValueError("No hospitalization columns found in the source schema.")
    if data.patid.isna().any() or data.patid.duplicated().any():
        raise ValueError("Expected one index record per patient and complete patient IDs.")
    index_date = pd.to_datetime(data.date, errors="coerce")
    registration = pd.to_datetime(data.regstartdate, errors="coerce")
    hf_type = data.HF_type.str.strip().replace("", pd.NA)
    rules = [
        ("At least one calendar year since registration",
         (index_date - registration).dt.days.gt(365)),
        ("At least one populated hospitalization field",
         data[hospitalization_columns].notna().any(axis=1)),
        ("Recorded heart failure type", hf_type.notna()),
    ]
    flow = [{"stage": "Patients in source dataset", "included": len(data), "excluded": 0}]
    eligible = pd.Series(True, index=data.index)
    for stage, passes in rules:
        previous = int(eligible.sum())
        eligible &= passes
        included = int(eligible.sum())
        flow.append(dict(stage=stage, included=included, excluded=previous - included))
    cohort = data.loc[eligible].copy()
    cohort["HF_type"] = hf_type.loc[eligible]
    return cohort, pd.DataFrame(flow)
