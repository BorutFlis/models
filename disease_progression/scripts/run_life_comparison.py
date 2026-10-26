"""Prepare the accepted cohort and run LIFE-Preserved batch assessments."""

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .life_preserved import (INPUT_IDS as LIFE_INPUT_IDS, RADIO_OPTIONS as LIFE_RADIO_OPTIONS,
        TREATMENTS as LIFE_TREATMENT_OPTIONS, attach_identifiers, create_driver,
        predict_profile, run_batch, fit_reference_defaults)
except ImportError:
    from life_preserved import (INPUT_IDS as LIFE_INPUT_IDS, RADIO_OPTIONS as LIFE_RADIO_OPTIONS,
        TREATMENTS as LIFE_TREATMENT_OPTIONS, attach_identifiers, create_driver,
        predict_profile, run_batch, fit_reference_defaults)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RESULTS_DIR = DATA_DIR / "results"

hosp_cols = [
    'pre_all_hosp_total_duration', 'pre_all_hosp_n',
    'post_all_hosp_total_duration', 'post_all_hosp_n',
    'pre_emmergency_hosp_total_duration', 'pre_emmergency_hosp_n',
    'post_emmergency_hosp_total_duration', 'post_emmergency_hosp_n',
    'pre_all_hosp_total_duration_in_30_days', 'pre_all_hosp_n_in_30_days',
    'pre_all_hosp_total_duration_in_60_days', 'pre_all_hosp_n_in_60_days',
    'pre_all_hosp_total_duration_in_90_days', 'pre_all_hosp_n_in_90_days',
    'post_all_hosp_total_duration_in_30_days', 'post_all_hosp_n_in_30_days',
    'post_all_hosp_total_duration_in_60_days', 'post_all_hosp_n_in_60_days',
    'post_all_hosp_total_duration_in_90_days', 'post_all_hosp_n_in_90_days',
    'pre_emmergency_hosp_total_duration_in_30_days',
    'pre_emmergency_hosp_n_in_30_days',
    'pre_emmergency_hosp_total_duration_in_60_days',
    'pre_emmergency_hosp_n_in_60_days',
    'pre_emmergency_hosp_total_duration_in_90_days',
    'pre_emmergency_hosp_n_in_90_days',
    'post_emmergency_hosp_total_duration_in_30_days',
    'post_emmergency_hosp_n_in_30_days',
    'post_emmergency_hosp_total_duration_in_60_days',
    'post_emmergency_hosp_n_in_60_days',
    'post_emmergency_hosp_total_duration_in_90_days',
    'post_emmergency_hosp_n_in_90_days', 'post_emmergency_days_to_hosp'
]


def prepare_cohort(data_dir=DATA_DIR):
    """Keep the accepted proxies and complete numeric case selection."""
    df = pd.read_csv(data_dir / "processed" / "high_risk_HES_big.csv", index_col=0)
    df = df.loc[df.loc[:, hosp_cols].notna().any(axis=1)].copy()
    life_mapping_df = attach_identifiers(pd.read_csv(
        data_dir / "processed" / "LIFE_Preserved_high_risk_HES_mapping.csv"))
    ischaemic_heart_disease_cols = life_mapping_df.loc[
        life_mapping_df["mapping_status"].eq("Derived proxy"),
        "high_risk_HES_schema_attribute",
    ].squeeze().split("|")
    ischaemic_heart_disease_cols = list(map(str.strip, ischaemic_heart_disease_cols))
    df["Ischemic heart disease"] = np.where(
        df.loc[:, ischaemic_heart_disease_cols].eq("Y").any(axis=1), "Y", "N"
    )

    matched = life_mapping_df["mapping_status"].isin(["Exact field", "Matched"])
    overlap_attrs = life_mapping_df.loc[matched, "high_risk_HES_schema_attribute"].to_list()
    non_na_attrs = life_mapping_df.loc[
        matched & life_mapping_df["schema_dtype"].eq("float64"),
        "high_risk_HES_schema_attribute",
    ].to_list()
    overlap_attrs.append("Ischemic heart disease")

    df["Previous HF hospitalization"] = np.where(
        df["pre_emmergency_hosp_n_in_90_days"].gt(1), "Y", "N"
    )

    X = df.dropna(subset=non_na_attrs).loc[:, overlap_attrs]
    X["Previous HF hospitalization"] = df.loc[X.index, "Previous HF hospitalization"]
    X_hfpef = X.loc[df.loc[X.index, "HF_type"].eq("hfpef")]
    return X_hfpef, life_mapping_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "life_preserved")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--hb-unit", choices=["gdL", "mmolL"], default="gdL")
    parser.add_argument("--defaults", type=Path, help="JSON of defaults keyed by website ID")
    parser.add_argument("--reference-csv", type=Path,
                        help="Reference cohort for numeric means and categorical modes; first column is row ID")
    parser.add_argument("--treatment", action="append", default=[], choices=list(LIFE_TREATMENT_OPTIONS.values()))
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    defaults = json.loads(args.defaults.read_text()) if args.defaults else None
    frame, mapping = prepare_cohort(args.data_dir)
    if args.reference_csv:
        fitted = fit_reference_defaults(pd.read_csv(args.reference_csv, index_col=0), mapping,
                                        hb_unit=args.hb_unit)
        defaults = {**fitted, **(defaults or {})}
    if args.limit is not None:
        if args.limit < 0:
            parser.error("--limit must be nonnegative")
        frame = frame.head(args.limit)
    driver = create_driver(headless=not args.headed)
    try:
        summary = run_batch(frame, mapping, partial(predict_profile, driver), args.output_dir,
                            defaults=defaults, hb_unit=args.hb_unit, treatments=args.treatment,
                            retries=args.retries, delay=args.delay, resume=not args.no_resume)
        print(f"Saved {len(summary)} rows to {args.output_dir / 'predictions.csv'}")
        if not summary.empty and summary.status.eq("error").any():
            raise SystemExit(1)
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
