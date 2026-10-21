"""Train three conditional HF classifiers and evaluate monotone cumulative risk.

Run: python -m early_diagnosis.scripts.conditional_risk.reproduce_experiments
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from abstract_models.imputation import median_imputer
from abstract_models.metric_utils import compute_binary_classification_metrics

PROJECT_DIR = Path(__file__).resolve().parents[2]
INTERVALS = ((0, 1), (1, 2), (2, 5))
LOGGER = logging.getLogger(__name__)


def predictor_columns(frame: pd.DataFrame) -> list[str]:
    """Allow only baseline clinical measurements, never outcome information."""
    return [name for name in frame if name.startswith(
        ("Blo_", "Phy_", "Pat_", "Sym_", "Med_"))]


def load_cohort(data_dir: Path, files: list[str], patients_per_file: int = 200,
                seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample patients, retaining all their eligible pre-diagnosis timepoints."""
    gather_baselines, gather_counts = [], []
    for filename in files:
        path = data_dir / filename
        frame = pd.read_csv(path, dtype={"ID": str}, low_memory=False)
        with (data_dir / "id_maps" / f"{path.stem}.json").open() as handle:
            id_map = json.load(handle)
        mapped = frame["ID"].map(id_map)
        if mapped.isna().any():
            raise ValueError(f"Unmapped patient identifiers in {filename}")
        frame["ID"] = mapped.astype(str)
        frame["Timepoint"] = pd.to_datetime(frame["Timepoint"], errors="raise")
        if frame[["ID", "Timepoint"]].isna().any().any():
            raise ValueError(f"Missing identifiers or dates in {filename}")
        frame = frame.sort_values(["ID", "Timepoint"])
        if frame.duplicated(["ID", "Timepoint"]).any():
            raise ValueError(f"Duplicate patient dates in {filename}")
        frame["days_to_HFD"] = pd.to_numeric(frame["days_to_HFD"], errors="raise")
        dates = frame["Timepoint"] + pd.to_timedelta(frame["days_to_HFD"], unit="D")
        first_event = dates.groupby(frame["ID"]).transform("min")
        # Do not label diagnosed patients with missing event timing as controls.
        ever_diagnosed = frame.groupby("ID")["Dia_HFD_patient"].transform("max").eq(1)
        unknown_event = ever_diagnosed & first_event.isna()
        last_date = frame.groupby("ID")["Timepoint"].transform("max")
        frame["followup_days"] = (last_date - frame["Timepoint"]).dt.days
        features = predictor_columns(frame)
        frame[features] = frame.groupby("ID", sort=False)[features].ffill()
        eligible = ~unknown_event & (
            first_event.isna() | frame["Timepoint"].lt(first_event))
        eligible_ids = frame.loc[eligible, "ID"].drop_duplicates().sort_values()
        if patients_per_file > len(eligible_ids):
            raise ValueError(f"{filename}: requested {patients_per_file} patients but only {len(eligible_ids)} eligible")
        selected_ids = eligible_ids.sample(n=patients_per_file, random_state=seed)
        baseline = frame.loc[eligible & frame["ID"].isin(selected_ids)].copy()
        baseline["days_to_HFD"] = (
            first_event.loc[baseline.index] - baseline["Timepoint"]
        ).dt.total_seconds() / 86400
        baseline["source_file"] = filename
        gather_baselines.append(baseline)
        gather_counts.append(dict(source_file=filename, raw_rows=len(frame),
            raw_patients=frame["ID"].nunique(), eligible_patients=len(eligible_ids),
            sampled_patients=baseline["ID"].nunique(), timepoints=len(baseline),
            patients_missing_event_timing=frame.loc[unknown_event, "ID"].nunique()))
        LOGGER.info("%s: %s sampled patients, %s timepoints from %s raw rows",
                    filename, baseline["ID"].nunique(), len(baseline), len(frame))
    cohort = pd.concat(gather_baselines, ignore_index=True)
    if cohort.empty or cohort.groupby("ID")["source_file"].nunique().gt(1).any():
        raise ValueError("Empty cohort or overlapping patient histories across files")
    return cohort, pd.DataFrame(gather_counts)


def patient_splits(cohort: pd.DataFrame, folds: int, seed: int):
    """Stratify unique patients and return row positions with disjoint patients."""
    patients = cohort.groupby("ID", sort=True)["days_to_HFD"].agg(lambda values: values.notna().any())
    if patients.value_counts().min() < folds:
        raise ValueError("Too few event/non-event patients for the requested folds")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train, test in splitter.split(patients.index, patients.astype(int)):
        train_rows = np.flatnonzero(cohort["ID"].isin(patients.index[train]).to_numpy())
        test_rows = np.flatnonzero(cohort["ID"].isin(patients.index[test]).to_numpy())
        if set(cohort.iloc[train_rows]["ID"]) & set(cohort.iloc[test_rows]["ID"]):
            raise AssertionError("Patient leakage between training and validation")
        yield train_rows, test_rows


def create_targets(cohort: pd.DataFrame) -> pd.DataFrame:
    """Intervals are (start, end]; exclude prior events and unknown outcomes."""
    event = cohort["days_to_HFD"]
    targets = pd.DataFrame(index=cohort.index)
    for start, end in INTERVALS:
        at_risk = event.isna() | event.gt(start * 365.25)
        positive = at_risk & event.le(end * 365.25)
        negative = at_risk & (event.gt(end * 365.25) | (
            event.isna() & cohort["followup_days"].ge(end * 365.25)))
        targets[f"conditional_{start}_{end}Y"] = np.nan
        targets.loc[negative, f"conditional_{start}_{end}Y"] = 0
        targets.loc[positive, f"conditional_{start}_{end}Y"] = 1
        targets[f"HF_{end}Y"] = np.nan
        targets.loc[event.gt(0) & event.le(end * 365.25), f"HF_{end}Y"] = 1
        targets.loc[event.gt(end * 365.25) | (event.isna() &
            cohort["followup_days"].ge(end * 365.25)), f"HF_{end}Y"] = 0
    return targets


def cumulative_risks(hazards: np.ndarray) -> np.ndarray:
    """Convert conditional probabilities to nondecreasing cumulative risks."""
    hazards = np.asarray(hazards, dtype=float)
    if hazards.ndim != 2 or hazards.shape[1] != 3:
        raise ValueError("Expected three conditional probabilities per patient")
    if not np.isfinite(hazards).all() or ((hazards < 0) | (hazards > 1)).any():
        raise ValueError("Conditional probabilities must be finite and in [0, 1]")
    return 1 - np.cumprod(1 - hazards, axis=1)


def predict_risks(models: dict, features: pd.DataFrame) -> pd.DataFrame:
    """Apply saved interval pipelines in chronological order to baseline rows."""
    hazards = np.column_stack([models[f"conditional_{start}_{end}Y"].predict_proba(
        features)[:, 1] for start, end in INTERVALS])
    return pd.DataFrame(cumulative_risks(hazards), index=features.index,
                        columns=["risk_1Y", "risk_2Y", "risk_5Y"])


def make_pipeline(args: argparse.Namespace) -> Pipeline:
    """Reuse package preprocessing; retain empty features consistently in folds."""
    preprocessor = clone(median_imputer)
    preprocessor.set_params(num__imputer__keep_empty_features=True,
        cat__imputer__strategy="constant", cat__imputer__fill_value="Missing",
        cat__imputer__keep_empty_features=True)
    return Pipeline([("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=args.n_estimators,
            min_samples_leaf=5, random_state=args.seed, n_jobs=args.n_jobs))])


def score_predictions(y: pd.Series, probability: np.ndarray, patient_ids: pd.Series) -> dict:
    """Score known outcomes only; probabilities are always out of fold."""
    known = y.notna().to_numpy()
    truth, probability = y.loc[known].astype(int), probability[known]
    if truth.nunique() < 2:
        raise ValueError("Evaluation requires both classes among known outcomes")
    metrics = compute_binary_classification_metrics(truth, probability >= 0.5, probability).to_dict()
    metrics.update(average_precision=average_precision_score(truth, probability),
        brier_score=brier_score_loss(truth, probability), timepoints=len(truth),
        patients=patient_ids.loc[known].nunique(), positive_timepoints=int(truth.sum()),
        unknown_or_not_at_risk_timepoints=int((~known).sum()))
    return metrics


def run(args: argparse.Namespace) -> None:
    cohort, counts = load_cohort(args.data_dir, args.files, args.patients_per_file, args.seed)
    targets = create_targets(cohort)
    target_container = [f"conditional_{start}_{end}Y" for start, end in INTERVALS]
    features = cohort[predictor_columns(cohort)].copy()
    numeric = [name for name in features if name.startswith(("Blo_", "Med_")) or (
        name.startswith("Phy_") and name != "Phy_Sex")]
    features[numeric] = features[numeric].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan)
    categorical = features.columns.difference(numeric)
    features[categorical] = features[categorical].astype(object).where(features[categorical].notna(), np.nan)
    hazards = np.full((len(cohort), 3), np.nan)
    fold_ids = np.full(len(cohort), -1)
    gather_fold_metrics, gather_importance, models = [], [], {}
    for fold, (train, test) in enumerate(patient_splits(cohort, args.folds, args.seed)):
        fold_ids[test] = fold
        for column, target in enumerate(target_container):
            train_known = targets.loc[train, target].dropna().index
            if targets.loc[train_known, target].nunique() != 2:
                raise ValueError(f"Fold {fold}, {target}: training requires both classes")
            pipeline = make_pipeline(args)
            pipeline.fit(features.loc[train_known], targets.loc[train_known, target].astype(int))
            hazards[test, column] = pipeline.predict_proba(features.iloc[test])[:, 1]
            gather_fold_metrics.append(dict(fold=fold, target=target,
                train_patients=cohort.loc[train_known, "ID"].nunique(),
                test_patients=cohort.iloc[test]["ID"].nunique(),
                train_timepoints=len(train_known),
                train_positive_timepoints=int(targets.loc[train_known, target].sum())))
        LOGGER.info("Finished OOF fold %s/%s", fold + 1, args.folds)
    risks = cumulative_risks(hazards)
    if (np.diff(risks, axis=1) < -1e-12).any() or (fold_ids < 0).any():
        raise AssertionError("OOF coverage or monotonicity failed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gather_metrics = []
    for column, (start, end) in enumerate(INTERVALS):
        target = target_container[column]
        gather_metrics.append(dict(kind="conditional", target=target,
            **score_predictions(targets[target], hazards[:, column], cohort["ID"])))
        metrics = score_predictions(targets[f"HF_{end}Y"], risks[:, column], cohort["ID"])
        gather_metrics.append(dict(kind="cumulative", target=f"HF_{end}Y", **metrics))
        LOGGER.info("%sY cumulative OOF AUROC %.3f (%s known timepoint outcomes)", end, metrics["AUC"], metrics["timepoints"])
        known = targets[target].dropna().index
        pipeline = make_pipeline(args)
        pipeline.fit(features.loc[known], targets.loc[known, target].astype(int))
        models[target] = pipeline
        importance = pd.DataFrame({"target": target,
            "feature": pipeline["preprocessor"].get_feature_names_out(),
            "importance": pipeline["classifier"].feature_importances_})
        gather_importance.append(importance.sort_values("importance", ascending=False))
    joblib.dump(models, args.output_dir / "conditional_models.joblib")
    output = cohort[["ID", "Timepoint", "source_file"]].join(targets)
    output["fold"] = fold_ids
    for column, (start, end) in enumerate(INTERVALS):
        output[f"probability_{start}_{end}Y"] = hazards[:, column]
        output[f"risk_{end}Y"] = risks[:, column]
    output.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(gather_metrics).to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame(gather_fold_metrics).to_csv(args.output_dir / "fold_counts.csv", index=False)
    pd.concat(gather_importance).to_csv(args.output_dir / "feature_importance.csv", index=False)
    counts.to_csv(args.output_dir / "cohort_counts.csv", index=False)
    processed = cohort.drop(columns=predictor_columns(cohort)).join(features).join(targets)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    processed.to_csv(args.processed_dir / "conditional_risk_timepoints.csv", index=False)
    schema = pd.DataFrame({"attr": processed.columns,
        "dtype": [str(dtype) for dtype in processed.dtypes],
        "variable_type": ["predictor" if name in features else "targets" if name in targets
                          else "meta" for name in processed]})
    schema.to_csv(args.processed_dir / "conditional_risk_timepoints_schema.csv", index=False)
    config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    config.update(predictors=features.columns.tolist(), target_container=target_container,
        patients=cohort["ID"].nunique(), timepoints=len(cohort), intervals=INTERVALS,
        baseline="every observed pre-diagnosis timepoint; NT measurement not required",
        metric_weighting="each eligible timepoint has equal weight",
        monotonicity_verified=True, validation="patient-level out-of-fold predictions")
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data/raw/cprd_NT")
    parser.add_argument("--files", nargs="+", default=["29.csv", "8.csv"])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs/conditional_risk_all_timepoints")
    parser.add_argument("--patients-per-file", type=int, default=200)
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_DIR / "data/processed")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.folds < 2 or args.n_estimators < 1 or args.patients_per_file < 1:
        parser.error("At least two folds, one estimator and one patient per file are required")
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())
