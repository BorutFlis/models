from pathlib import Path

p = Path('disease_progression/scripts/run_life_comparison.py')
s = p.read_text()
hosp = s[s.index('hosp_cols ='):s.index('df = pd.read_csv')]
cohort = s[s.index('ischaemic_heart_disease_cols ='):s.index('# use Selenium')]
cohort = cohort.replace('schema_df = pd.read_csv(os.path.join(DATA_DIR, "processed", "high_risk_HES_big_schema.csv"))', '')
cohort = cohort.replace('X = pd.concat([X, df.loc[X.index, "pre_emmergency_hosp_n_in_90_days"].gt(1)], axis=1)',
                        'X["Previous HF hospitalization"] = df.loc[X.index, "Previous HF hospitalization"]')
start = '''"""Prepare the accepted cohort and run LIFE-Preserved batch assessments."""

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .life_preserved import (INPUT_IDS as LIFE_INPUT_IDS, RADIO_OPTIONS as LIFE_RADIO_OPTIONS,
        TREATMENTS as LIFE_TREATMENT_OPTIONS, attach_identifiers, create_driver,
        predict_profile, run_batch)
except ImportError:
    from life_preserved import (INPUT_IDS as LIFE_INPUT_IDS, RADIO_OPTIONS as LIFE_RADIO_OPTIONS,
        TREATMENTS as LIFE_TREATMENT_OPTIONS, attach_identifiers, create_driver,
        predict_profile, run_batch)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RESULTS_DIR = DATA_DIR / "results"

'''
body = '''df = pd.read_csv(data_dir / "processed" / "high_risk_HES_big.csv", index_col=0)
df = df.loc[df.loc[:, hosp_cols].notna().any(axis=1)].copy()
life_mapping_df = attach_identifiers(pd.read_csv(
    data_dir / "processed" / "LIFE_Preserved_high_risk_HES_mapping.csv"))
''' + cohort + 'return X_hfpef, life_mapping_df\n'
cli = '''
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "life_preserved")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--hb-unit", choices=["gdL", "mmolL"], default="gdL")
    parser.add_argument("--defaults", type=Path, help="JSON of defaults keyed by website ID")
    parser.add_argument("--treatment", action="append", default=[], choices=list(LIFE_TREATMENT_OPTIONS.values()))
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    import json
    defaults = json.loads(args.defaults.read_text()) if args.defaults else None
    frame, mapping = prepare_cohort(args.data_dir)
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
'''
p.write_text(start + hosp + 'def prepare_cohort(data_dir=DATA_DIR):\n'
             + '    """Keep the accepted proxies and complete numeric case selection."""\n'
             + ''.join('    ' + line + '\n' for line in body.splitlines()) + cli, encoding='utf-8')
