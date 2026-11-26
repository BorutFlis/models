"""Cross-validate the high-risk classifier and compare mortality rankings."""

import json
from pathlib import Path
import sys

from lifelines.utils import concordance_index
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT.parent))
from abstract_models.imputation import median_imputer
from abstract_models.metric_utils import compute_binary_classification_metrics

PROBABILITY = "high_risk_probability"
BASELINE = "death_1_Y"
OUTCOMES = ["death_event", "days_to_event"]


def classification_metrics(y, probability):
    metrics = compute_binary_classification_metrics(y, probability > 0.5, probability)
    metrics["Average Precision"] = average_precision_score(y, probability)
    metrics["Brier Score"] = brier_score_loss(y, probability)
    metrics["Log Loss"] = log_loss(y, probability)
    metrics["n_patients"] = len(y)
    metrics["n_positive"] = y.sum()
    return metrics


def save_feature_summary(features, y, categorical, output):
    numeric = features.drop(columns=categorical)
    numeric_summary = (
        numeric.groupby(y).quantile([0.25, 0.5, 0.75]).stack(future_stack=True)
        .unstack(level=1).rename(columns={0.25: "q25", 0.5: "median", 0.75: "q75"})
        .rename_axis(index=[y.name, "feature"]).reset_index()
    )
    numeric_summary.to_csv(output / "numeric_features_by_class.csv", index=False)

    gather_distributions = []
    for feature in categorical:
        counts = (
            features[feature].groupby(y).value_counts(dropna=False)
            .rename("count").rename_axis([y.name, "value"]).reset_index()
        )
        counts["fraction"] = counts["count"] / counts[y.name].map(y.value_counts())
        gather_distributions.append(counts.assign(feature=feature))
    pd.concat(gather_distributions, ignore_index=True).to_csv(
        output / "categorical_features_by_class.csv", index=False
    )


def compare_mortality(test, baseline_path, output):
    baseline = pd.read_csv(
        baseline_path, usecols=["patid", BASELINE], dtype={"patid": str}
    )
    if baseline["patid"].isna().any() or baseline["patid"].duplicated().any():
        raise ValueError("Baseline must have one prediction row per nonmissing patid.")
    comparison = test.merge(baseline, on="patid", how="left", validate="one_to_one")
    eligible = (
        np.isfinite(comparison[BASELINE])
        & np.isfinite(comparison["days_to_event"])
        & comparison["death_event"].isin([0, 1])
    )
    comparison["included_in_cindex"] = eligible
    comparison.to_csv(output / "mortality_comparison.csv", index=False)
    matched = comparison.loc[eligible]
    scores = {
        column: concordance_index(
            matched["days_to_event"], -matched[column], matched["death_event"]
        )
        for column in [PROBABILITY, BASELINE]
    }
    metrics = {
        "n_test": len(test),
        "n_matched_patids": int(test["patid"].isin(baseline["patid"]).sum()),
        "n_evaluated": len(matched),
        "n_deaths": int(matched["death_event"].sum()),
        "c_index": scores,
        "c_index_difference": scores[PROBABILITY] - scores[BASELINE],
    }
    (output / "mortality_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    target = config["target"]
    output = PROJECT / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)

    schema = pd.read_csv(PROJECT / config["schema"])
    predictors = schema.loc[schema["variable_type"].eq("predictor"), "attr"].tolist()
    categorical = schema.loc[schema["attr"].isin(predictors) & schema["dtype"].eq("object"), "attr"].tolist()
    identity = ["patid", "date"]
    excluded = {*identity, *OUTCOMES, target}
    if excluded.intersection(predictors) or any(name.startswith("post_") for name in predictors):
        raise ValueError("The predictor schema includes identifiers or outcome information.")
    dtypes = {name: str if name in categorical else np.float32 for name in predictors}
    data = pd.read_csv(
        PROJECT / config["dataset"], usecols=[*identity, *OUTCOMES, target, *predictors],
        dtype={**dtypes, "patid": str},
    )
    if data["patid"].isna().any() or data["patid"].duplicated().any():
        raise ValueError("The cohort must have one row per nonmissing patid.")

    labeled = data[target].notna()
    X = data.loc[labeled, predictors]
    y = data.loc[labeled, target]
    if set(y.unique()) != {0, 1} or y.value_counts().min() < config["n_splits"]:
        raise ValueError("Each target class must have at least one patient per fold.")
    y = y.astype(int)

    seed = config["random_state"]
    cv = StratifiedKFold(n_splits=config["n_splits"], shuffle=True, random_state=seed)
    pipeline = make_pipeline(
        clone(median_imputer), RandomForestClassifier(random_state=seed, **config["random_forest"])
    )
    oof = data.loc[labeled, [*identity, target]].copy()
    oof["fold"] = 0
    oof[PROBABILITY] = np.nan
    gather_metrics = {}
    for fold, (train, validation) in enumerate(cv.split(X, y), start=1):
        print(f"Training fold {fold}/{cv.n_splits}", flush=True)
        model = clone(pipeline).fit(X.iloc[train], y.iloc[train])
        probability = model.predict_proba(X.iloc[validation])[:, 1]
        oof.loc[X.index[validation], "fold"] = fold
        oof.loc[X.index[validation], PROBABILITY] = probability
        gather_metrics[fold] = classification_metrics(y.iloc[validation], probability)

    oof.to_csv(output / "oof_predictions.csv", index=False)
    fold_metrics = pd.DataFrame.from_dict(gather_metrics, orient="index").rename_axis("fold")
    fold_metrics.to_csv(output / "cv_metrics.csv")
    summary = pd.DataFrame({
        "fold_mean": fold_metrics.mean(),
        "fold_std": fold_metrics.std(),
        "pooled_oof": classification_metrics(y, oof[PROBABILITY]),
    })
    summary.to_csv(output / "cv_summary.csv", index_label="metric")

    print(f"Refitting on all {len(y):,} labeled patients", flush=True)
    pipeline.fit(X, y)
    preprocessor, forest = pipeline.steps[0][1], pipeline.steps[1][1]
    pd.Series(
        forest.feature_importances_, index=preprocessor.get_feature_names_out(), name="importance"
    ).sort_values(ascending=False).to_csv(output / "feature_importance.csv", index_label="feature")
    save_feature_summary(X, y, categorical, output)

    test = data.loc[~labeled, [*identity, *OUTCOMES]].copy()
    test[PROBABILITY] = np.nan
    batch_size = config["prediction_batch_size"]
    for start in range(0, len(test), batch_size):
        batch = test.index[start:start + batch_size]
        test.loc[batch, PROBABILITY] = pipeline.predict_proba(data.loc[batch, predictors])[:, 1]
    test[[*identity, PROBABILITY]].to_csv(output / "test_predictions.csv", index=False)
    compare_mortality(test, PROJECT / config["baseline"], output)


if __name__ == "__main__":
    main()
