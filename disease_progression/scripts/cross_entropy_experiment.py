"""Cross-validate exposure-weighted LightGBM across L2 regularization values.

Run from the repository root with:
    python -m disease_progression.scripts.cross_entropy_experiment

``cross_entropy_lambda`` is an objective, not a tunable scalar. The sweep
below varies ``reg_lambda``. Its learned intensity is converted to probability
at unit exposure, without using test outcomes to construct prediction weights.
These binary metrics describe observed death_event, not fixed-horizon mortality.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sksurv.exceptions import NoComparablePairException
from sksurv.metrics import concordance_index_censored

# Allow both direct script execution and python -m from the repository root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from abstract_models.imputation import median_imputer


lambda_container = [0.1, 5, 10]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TARGET = "death_event"
EXPOSURE = "days_to_event"
attr_selection = [
    "Sym_DAR__ever", "Sym_DAR__in_last_year", "Sym_DAR__count",
    "Pat_DM__ever", "Pat_DM__in_last_year", "Pat_DM__count",
    "Pat_Hyp__ever", "Pat_Hyp__in_last_year", "Pat_Hyp__count",
    "Phy_Sex", "Phy_Age", "Phy_Sys__median_1.5Y", "Phy_Sys__last",
    "Phy_Dia__last", "Phy_Dia__median_1.5Y", "Blo_Hb__median_1.5Y",
    "Blo_Hb__last", "Blo_Cre__median_1.5Y", "Blo_Cre__last",
]


def exposure_weights(y, duration):
    """Reproduce adjusted exposure using training data only."""
    weights = pd.Series(index=y.index, dtype=float)
    for label in (0, 1):
        mask = y.eq(label)
        maximum = duration.loc[mask].max()
        if maximum <= 0:
            raise ValueError(f"Class {label} needs a positive maximum days_to_event.")
        normalized = duration.loc[mask] / maximum
        weights.loc[mask] = normalized if label == 0 else 1 - normalized
    return weights.clip(lower=0.001)


def binary_metrics(y, probability, duration, risk, threshold=0.5):
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    try:
        cindex = concordance_index_censored(
            np.asarray(y, dtype=bool), np.asarray(duration), risk,
        )[0]
    except NoComparablePairException:
        cindex = np.nan
    return {
        "Accuracy": accuracy_score(y, prediction),
        "Balanced Accuracy": balanced_accuracy_score(y, prediction),
        "AUC": roc_auc_score(y, probability),
        "Average Precision": average_precision_score(y, probability),
        "Precision": precision_score(y, prediction, zero_division=0),
        "Sensitivity (Recall)": recall_score(y, prediction, zero_division=0),
        "Specificity": tn / (tn + fp) if tn + fp else np.nan,
        "NPV": tn / (tn + fn) if tn + fn else np.nan,
        "F1 Score": f1_score(y, prediction, zero_division=0),
        "MCC": matthews_corrcoef(y, prediction),
        "Log Loss": log_loss(y, probability, labels=[0, 1]),
        "Brier Score": brier_score_loss(y, probability),
        "cindex": cindex,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def run_experiment(df, lambdas=lambda_container, n_splits=5, n_estimators=500,
                   threshold=0.5, objective="cross_entropy_lambda"):
    """Return fold metrics and mean/sample-standard-deviation summaries."""
    if objective not in ("cross_entropy_lambda", "cross_entropy"):
        raise ValueError("Unsupported objective.")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    lambdas = list(lambdas)
    if not lambdas or not np.isfinite(lambdas).all() or min(lambdas) < 0:
        raise ValueError("Provide finite, nonnegative reg_lambda values.")
    if n_splits < 2 or n_estimators < 1:
        raise ValueError("Need at least two folds and one estimator.")
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("death_event must contain only 0 and 1, with no missing values.")
    duration = pd.to_numeric(df[EXPOSURE], errors="raise")
    if not np.isfinite(duration).all() or (duration < 0).any():
        raise ValueError("days_to_event must be finite and nonnegative.")
    y = df[TARGET].astype(int)
    if y.nunique() != 2 or y.value_counts().min() < n_splits:
        raise ValueError("Each class must have at least n_splits observations.")
    X = df.loc[:, attr_selection]
    splits = list(StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=42,
    ).split(X, y))
    rows = []
    for reg_lambda in lambdas:
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            model = Pipeline([
                ("preprocessor", clone(median_imputer)),
                ("classifier", LGBMClassifier(
                    objective=objective, reg_lambda=reg_lambda,
                    n_estimators=n_estimators, learning_rate=0.03,
                    num_leaves=31, random_state=42, verbosity=-1, n_jobs=4,
                )),
            ])
            weights = exposure_weights(y.iloc[train_idx], duration.iloc[train_idx])
            model.fit(X.iloc[train_idx], y.iloc[train_idx])

            # Use raw scores: predict_proba for xentlambda returns intensities.
            raw_score = model.predict_proba(X.iloc[test_idx], raw_score=True)
            intensity = np.logaddexp(0, raw_score)
            # p = 1 - exp(-intensity * exposure); evaluation exposure is 1.
            probability = -np.expm1(-intensity)
            metrics = binary_metrics(
                y.iloc[test_idx], probability, duration.iloc[test_idx],
                raw_score, threshold,
            )
            rows.append({"objective": objective, "reg_lambda": reg_lambda,
                         "fold": fold, "threshold": threshold, **metrics})
            print(f"reg_lambda={reg_lambda:g}, fold={fold}/{n_splits}: "
                  f"AUC={metrics['AUC']:.4f}, cindex={metrics['cindex']:.4f}",
                  flush=True)
    results = pd.DataFrame(rows)
    metric_columns = list(metrics)
    summary = results.groupby(["objective", "reg_lambda", "threshold"])[
        metric_columns
    ].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return results, summary.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=DATA_DIR / "processed" / "high_risk_HES_big.csv")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "results")
    parser.add_argument("--lambdas", type=float, nargs="+", default=lambda_container,
                        help="L2 regularization values (reg_lambda).")
    parser.add_argument("--objective", choices=["cross_entropy_lambda", "cross_entropy"],
                        default="cross_entropy_lambda")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    df = pd.read_csv(args.data, usecols=attr_selection + [TARGET, EXPOSURE])
    results, summary = run_experiment(
        df, args.lambdas, args.n_splits, args.n_estimators,
        args.threshold, args.objective,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, table in (("folds", results), ("summary", summary)):
        path = args.output_dir / f"{args.objective}_experiment_{suffix}.csv"
        table.to_csv(path, index=False)
        print(f"Saved {path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
