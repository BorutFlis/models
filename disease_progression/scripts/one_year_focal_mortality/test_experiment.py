"""Check cohort eligibility, reported counts, and cross-experiment alignment."""

import numpy as np
import pandas as pd
import pytest

from prepare_cohort import resolve_source
from utils import CONFIG, IDENTITY, OUTCOMES, PROJECT, SCORE, TARGET


OUTPUT = PROJECT / CONFIG["output"]
PATIENT_TYPES = dict.fromkeys(IDENTITY, str)
COUNTS = ["n_held_out", "n_labeled", "n_deaths_within_1_year"]


@pytest.fixture(scope="module")
def cohort():
    return pd.read_parquet(PROJECT / CONFIG["cohort"], columns=[*IDENTITY, *OUTCOMES, TARGET, "HF_type"])


@pytest.fixture(scope="module")
def oof():
    return pd.read_csv(OUTPUT / "oof_predictions.csv", dtype=PATIENT_TYPES)


@pytest.fixture
def source_eligibility():
    schema = pd.read_csv(resolve_source(PROJECT / CONFIG["schema"]))
    hosp = [name for name in schema.attr if name.startswith(("pre_", "post_")) and "hosp" in name]
    raw = pd.read_csv(
        resolve_source(PROJECT / CONFIG["dataset"]),
        usecols=[*IDENTITY, *OUTCOMES, "regstartdate", *hosp],
        dtype={**PATIENT_TYPES, "regstartdate": str}, skipinitialspace=True, low_memory=False,
    )
    registration = pd.to_datetime(raw.regstartdate.str.strip(), errors="coerce")
    days = (pd.to_datetime(raw.date, errors="coerce") - registration).dt.days
    registered = raw.loc[days.gt(CONFIG["min_registration_days_exclusive"])]
    eligible = registered.dropna(subset=hosp, how="all")
    return eligible[[*IDENTITY, *OUTCOMES]].reset_index(drop=True), [len(raw), len(registered), len(eligible)]


@pytest.fixture
def expected_targets(cohort):
    duration, event = cohort.days_to_event, cohort.death_event
    valid = np.isfinite(duration) & duration.ge(0) & event.isin([0, 1])
    late = duration.gt(CONFIG["horizon_days"])
    target = np.select([valid & late, valid & ~late & event.eq(1)], [0.0, 1.0], default=np.nan)
    return cohort[[*IDENTITY, *OUTCOMES]].assign(**{TARGET: target}).set_index("patid").sort_index()


@pytest.fixture
def reported_counts(oof):
    by_fold = oof.groupby("fold")[TARGET].agg(["size", "count", "sum"]).set_axis(COUNTS, axis=1)
    folds = pd.read_csv(OUTPUT / "cv_metrics.csv", index_col="fold")[COUNTS]
    summaries = [pd.read_csv(path, index_col="metric").loc[COUNTS, "pooled_oof"]
                 for path in sorted(OUTPUT.glob("cv_summary*.csv"))]
    expected = by_fold.reindex(range(1, CONFIG["folds"] + 1))
    for number, summary in enumerate(summaries, start=1):
        folds.loc[f"pooled_{number}"] = summary
        expected.loc[f"pooled_{number}"] = by_fold.sum()
    return folds, expected


@pytest.fixture
def comparison_frames(oof):
    source = pd.read_csv(PROJECT / CONFIG["comparison"], dtype=PATIENT_TYPES)
    saved = pd.read_csv(OUTPUT / "mortality_comparison.csv", dtype=PATIENT_TYPES)
    expected = source.merge(oof[[*IDENTITY, SCORE]], how="left", on=IDENTITY, validate="one_to_one")
    expected["included_in_focal_comparison"] = source.included_in_cindex & expected[SCORE].notna()
    return saved[expected.columns], expected


@pytest.fixture
def hf_cohorts(cohort):
    hf = pd.read_parquet(PROJECT / "data/processed/hf_type_mortality_cohort.parquet", columns=[*IDENTITY, *OUTCOMES])
    focal = cohort.loc[cohort.HF_type.notna(), [*IDENTITY, *OUTCOMES]]
    return [frame.set_index(IDENTITY, verify_integrity=True).sort_index() for frame in (hf, focal)]


def test_cohort_matches_source(cohort, source_eligibility):
    expected, counts = source_eligibility
    pd.testing.assert_frame_equal(cohort[expected.columns], expected)
    assert pd.read_csv(OUTPUT / "cohort_flow.csv").included.tolist() == counts


def test_predictions_and_metric_counts(oof, expected_targets, reported_counts):
    assert oof.patid.is_unique and oof[SCORE].between(0, 1).all() and set(oof.fold) == set(range(1, CONFIG["folds"] + 1))
    pd.testing.assert_frame_equal(oof.set_index("patid")[expected_targets.columns].sort_index(), expected_targets)
    pd.testing.assert_frame_equal(*reported_counts, check_dtype=False)


def test_cross_experiment_records(comparison_frames, hf_cohorts):
    pd.testing.assert_frame_equal(*comparison_frames)
    hf, focal = hf_cohorts
    pd.testing.assert_frame_equal(hf, focal, obj="HF-type cohort")
