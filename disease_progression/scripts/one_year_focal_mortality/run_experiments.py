"""Run five-fold LightGBM focal-loss CV and compare survival concordance."""

import json

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from lightgbm import LGBMClassifier
from scipy.special import expit, log_expit
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, log_loss, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from utils import CONFIG, IDENTITY, OUTCOMES, PROJECT, SCORE, TARGET


def focal_objective(y, margin):
    """Derivatives of -alpha_t * (1 - p_t)**gamma * log(p_t)."""
    alpha = CONFIG["focal_alpha"]
    gamma = CONFIG["focal_gamma"]
    sign = 2 * y - 1
    signed_margin = sign * margin
    pt = expit(signed_margin)
    qt = expit(-signed_margin)
    log_pt = log_expit(signed_margin)
    weight = np.where(y == 1, alpha, 1 - alpha) * qt**gamma
    gradient = sign * weight * (gamma * pt * log_pt - qt)
    hessian = weight * pt * (
        gamma * (qt - gamma * pt) * log_pt + (2 * gamma + 1) * qt
    )
    # Focal loss can have negative curvature. LightGBM needs positive Hessians.
    return gradient, np.maximum(hessian, CONFIG["minimum_hessian"])


def classification_metrics(y, score):
    predicted = score >= CONFIG["classification_threshold"]
    return {
        "n_labeled": len(y),
        "n_deaths_within_1_year": int(y.sum()),
        "roc_auc": roc_auc_score(y, score),
        "average_precision": average_precision_score(y, score),
        "brier_score": brier_score_loss(y, score),
        "log_loss": log_loss(y, score),
        "sensitivity": recall_score(y, predicted, pos_label=1),
        "specificity": recall_score(y, predicted, pos_label=0),
        "f1": f1_score(y, predicted),
    }


def cross_validate():
    cohort_path = PROJECT / CONFIG["cohort"]
    schema = pd.read_csv(cohort_path.with_name(f"{cohort_path.stem}_schema.csv"))
    predictors = schema.loc[schema["variable_type"].eq("predictor"), "attr"].tolist()
    data = pd.read_parquet(cohort_path, columns=[*IDENTITY, *OUTCOMES, TARGET, *predictors])
    categorical = set(schema.loc[schema["dtype"].isin(["object", "str", "string"]), "attr"])
    dtypes = {name: "category" if name in categorical else np.float32 for name in predictors}
    X = data[predictors].astype(dtypes)
    target = data[TARGET]
    labeled = target.notna().to_numpy()
    if set(target[labeled].unique()) != {0, 1} or target.value_counts().min() < CONFIG["folds"]:
        raise ValueError("Each validation fold needs both target classes.")
    oof = data[[*IDENTITY, *OUTCOMES, TARGET]].assign(fold=0, **{SCORE: np.nan})
    del data
    cv = StratifiedKFold(n_splits=CONFIG["folds"], shuffle=True, random_state=CONFIG["seed"])
    print(f"{len(X):,} patients, {labeled.sum():,} known one-year outcomes", flush=True)
    for fold, (train, validation) in enumerate(cv.split(X, target.fillna(-1)), 1):
        print(f"Training fold {fold}/{cv.n_splits}", flush=True)
        train = train[labeled[train]]
        model = LGBMClassifier(
            objective=focal_objective, random_state=CONFIG["seed"], **CONFIG["lightgbm"],
        )
        model.fit(X.iloc[train], target.iloc[train].astype(int))
        score = expit(model.predict(X.iloc[validation], raw_score=True))
        oof.loc[validation, "fold"] = fold
        oof.loc[validation, SCORE] = score

    if not np.isfinite(oof[SCORE]).all() or oof["fold"].eq(0).any():
        raise ValueError("Every patient must receive one finite held-out score.")
    return oof


def save_cv_metrics(oof, output):
    gather_fold_metrics = []
    for fold, held_out in oof.groupby("fold"):
        known = held_out.dropna(subset=[TARGET])
        gather_fold_metrics.append({
            "fold": fold, "n_held_out": len(held_out),
            **classification_metrics(known[TARGET], known[SCORE]),
        })
    fold_metrics = pd.DataFrame(gather_fold_metrics).set_index("fold")
    known = oof.dropna(subset=[TARGET])
    summary = pd.DataFrame({
        "fold_mean": fold_metrics.mean(),
        "fold_std": fold_metrics.std(),
        "pooled_oof": {
            "n_held_out": len(oof),
            **classification_metrics(known[TARGET], known[SCORE]),
        },
    })
    fold_metrics.to_csv(output / "cv_metrics.csv")
    summary.to_csv(output / "cv_summary.csv", index_label="metric")


def compare_mortality(oof, output):
    source = pd.read_csv(
        PROJECT / CONFIG["comparison"],
        dtype={"patid": str, "date": str},
    )
    merged = source.merge(
        oof[[*IDENTITY, *OUTCOMES, TARGET, "fold", SCORE]],
        on=IDENTITY, how="left", validate="one_to_one",
        suffixes=("", "_focal"),
    )
    matched = merged["fold"].notna()
    for col in OUTCOMES:
        left, right = merged.loc[matched, [col, col + "_focal"]].to_numpy().T
        if not np.array_equal(left, right, equal_nan=True):
            raise ValueError(f"Joined patients disagree on {col}.")
    merged = merged.drop(columns=[col + "_focal" for col in OUTCOMES])

    scores = [SCORE, "death_1_Y", "high_risk_probability"]
    eligible = (
        merged["included_in_cindex"].eq(True)
        & np.isfinite(merged[[*scores, "days_to_event"]]).all(axis=1)
        & merged["death_event"].isin([0, 1])
        & merged["days_to_event"].ge(0)
    )
    merged["included_in_focal_comparison"] = eligible
    cohort = merged.loc[eligible]
    c_index = {
        name: concordance_index(cohort["days_to_event"], -cohort[name], cohort["death_event"])
        for name in scores
    }
    metrics = {
        "n_source": len(source),
        "n_matched_patient_dates": int(matched.sum()),
        "n_originally_eligible": int(source["included_in_cindex"].eq(True).sum()),
        "n_evaluated": len(cohort),
        "n_deaths": int(cohort["death_event"].sum()),
        "n_unknown_one_year_outcomes": int(cohort[TARGET].isna().sum()),
        "c_index": c_index,
        "focal_minus_baseline": {name: c_index[SCORE] - c_index[name] for name in scores[1:]},
    }
    merged.to_csv(output / "mortality_comparison.csv", index=False)
    (output / "mortality_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    output = PROJECT / CONFIG["output"]
    output.mkdir(parents=True, exist_ok=True)
    oof = cross_validate()
    oof.to_csv(output / "oof_predictions.csv", index=False)
    save_cv_metrics(oof, output)
    compare_mortality(oof, output)
    (output / "configuration.json").write_text(json.dumps(CONFIG, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
