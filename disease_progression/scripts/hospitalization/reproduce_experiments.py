"""Reproducible Random Forest baselines for every schema hospitalization target."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from abstract_models.imputation import median_imputer


def metrics(y, prediction):
    return dict(mae=mean_absolute_error(y, prediction),
                rmse=float(np.sqrt(mean_squared_error(y, prediction))),
                r2=r2_score(y, prediction))


def run(args):
    if args.folds < 2 or args.trees < 1:
        raise ValueError('Use at least two folds and one tree.')
    if args.max_rows is not None and args.max_rows < args.folds * 2:
        raise ValueError('max-rows must allow at least two rows per fold.')
    df = pd.read_csv(args.data, low_memory=False, dtype={'patid': str})
    schema = pd.read_csv(args.schema)
    if schema.attr.duplicated().any():
        raise ValueError('Schema contains duplicate attributes.')
    target_container = schema.loc[
        schema.variable_type.eq('targets') & schema.attr.str.startswith('post_')
        & schema.attr.str.contains('hosp'), 'attr'].tolist()
    predictors = schema.loc[schema.variable_type.eq('predictor'), 'attr'].tolist()
    if not target_container or not predictors:
        raise ValueError('Schema must define hospitalization targets and predictors.')
    missing = set(target_container + predictors + ['patid']) - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if any(c.startswith('post_') for c in predictors):
        raise ValueError('Future hospitalization outcomes cannot be predictors.')
    if df.patid.isna().any():
        raise ValueError('Patient IDs must be complete.')
    total_rows = len(df)
    # Same have_hosp_stats_filter as weighted_samples_experiment.py.
    hosp_cols = [c for c in df if c.startswith(('pre_', 'post_')) and 'hosp' in c]
    have_hosp_stats_filter = df.loc[:, hosp_cols].notna().any(axis=1)
    df = df.loc[have_hosp_stats_filter]
    filtered_rows = len(df)
    print(f'Hospitalization availability filter: {total_rows} -> {filtered_rows} rows', flush=True)
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=args.seed).sort_index()
    X_all = df[predictors].copy()
    # CSV inference can mix strings and numbers in categorical columns.
    categorical = schema.loc[schema.dtype.eq('object') & schema.attr.isin(predictors), 'attr']
    for col in predictors:
        if col in categorical.values:
            X_all[col] = X_all[col].map(lambda v: str(v) if pd.notna(v) else np.nan)
        else:
            X_all[col] = pd.to_numeric(X_all[col], errors='raise')
    X_all = X_all.replace([np.inf, -np.inf], np.nan)
    args.output.mkdir(parents=True, exist_ok=True)
    gather_summary = []
    gather_status = []
    for target in target_container:
        y_all = pd.to_numeric(df[target], errors='raise')
        valid = y_all.notna() & np.isfinite(y_all)
        X, y = X_all.loc[valid], y_all.loc[valid]
        groups = df.loc[valid, 'patid']
        status = dict(target=target, n_observed=len(y), n_missing=int((~valid).sum()))
        if len(y) < args.folds * 2 or groups.nunique() < args.folds or y.nunique() < 2:
            gather_status.append(dict(**status, status='skipped: insufficient data or constant target'))
            continue
        splitter = (GroupKFold(args.folds) if groups.duplicated().any()
                    else KFold(args.folds, shuffle=True, random_state=args.seed))
        splits = list(splitter.split(X, y, groups))
        destination = args.output / target
        destination.mkdir(exist_ok=True)
        predictions = pd.DataFrame({'source_row': X.index, 'patid': groups.to_numpy(),
                                    'observed': y.to_numpy()})
        gather_folds = []
        models = {'DummyMean': DummyRegressor(), 'RandomForest': RandomForestRegressor(
            n_estimators=args.trees, min_samples_leaf=2, max_depth=16,
            random_state=args.seed, n_jobs=args.jobs)}
        for name, estimator in models.items():
            pipeline = Pipeline([('preprocessor', clone(median_imputer)), ('model', estimator)])
            oof = np.full(len(y), np.nan)
            for fold, (train, test) in enumerate(splits, 1):
                print(f'{target}: {name}, fold {fold}/{args.folds}', flush=True)
                fitted = clone(pipeline).fit(X.iloc[train], y.iloc[train])
                oof[test] = fitted.predict(X.iloc[test])
                predictions.loc[test, 'fold'] = fold
                gather_folds.append(dict(model=name, fold=fold, n_train=len(train),
                                         n_test=len(test), **metrics(y.iloc[test], oof[test])))
            gather_summary.append(dict(target=target, model=name, n_observed=len(y),
                                       **metrics(y, oof)))
            predictions[name] = oof
            if name == 'RandomForest':
                # Full-data fit is for descriptive importance only, after OOF evaluation.
                pipeline.fit(X, y)
                pd.DataFrame({'feature': pipeline['preprocessor'].get_feature_names_out(),
                              'importance': pipeline['model'].feature_importances_}).sort_values(
                                  'importance', ascending=False).to_csv(
                                      destination / 'feature_importance.csv', index=False)
        pd.DataFrame(gather_folds).to_csv(destination / 'fold_metrics.csv', index=False)
        if len(df) < 10000 or args.save_oof:
            predictions.to_csv(destination / 'oof_predictions.csv', index=False)
        gather_status.append(dict(**status, status='completed'))
        pd.DataFrame(gather_summary).to_csv(args.output / 'summary.csv', index=False)
        pd.DataFrame(gather_status).to_csv(args.output / 'target_status.csv', index=False)
    pd.DataFrame(gather_status).to_csv(args.output / 'target_status.csv', index=False)
    configuration = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    configuration.update(total_source_rows=total_rows, hospitalization_filter_rows=filtered_rows,
                         hospitalization_filter_columns=hosp_cols, experiment_rows=len(df),
                         target_container=target_container, predictors=predictors,
                         sklearn_version=sklearn.__version__,
                         data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                         schema_sha256=hashlib.sha256(args.schema.read_bytes()).hexdigest())
    (args.output / 'configuration.json').write_text(json.dumps(configuration, indent=2))
    print(f'Results: {args.output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    data_dir = ROOT / 'disease_progression/data'
    parser.add_argument('--data', type=Path, default=data_dir / 'processed/high_risk_HES_big.csv')
    parser.add_argument('--schema', type=Path, default=data_dir / 'processed/high_risk_HES_big_schema.csv')
    parser.add_argument('--output', type=Path, default=data_dir / 'results/hospitalization')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--trees', type=int, default=100)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-rows', type=int, help='Seeded sample for smoke checks only.')
    parser.add_argument('--save-oof', action='store_true')
    run(parser.parse_args())
