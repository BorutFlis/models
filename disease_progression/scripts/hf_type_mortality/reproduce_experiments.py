"""Evaluate fixed-horizon mortality and compare age-adjusted HF type and sex groups."""

from itertools import combinations
import json
import sys

import lifelines
from lifelines.statistics import proportional_hazard_test
import numpy as np
import pandas as pd
from scipy.stats import norm
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from prepare_cohort import prepare
from utils import CONFIG, PROJECT

sys.path.insert(0, str(PROJECT.parent))
from abstract_models.imputation import median_imputer


def gather_descriptives(features, target, years):
    continuous, categorical = [], []
    for label, rows in features.groupby(target):
        numbers = rows.select_dtypes(include="number")
        quartiles = numbers.quantile([0.25, 0.50, 0.75]).T
        quartiles.columns = ["q25", "median", "q75"]
        continuous.append(quartiles.assign(years=years, death=label,
                                           observed=numbers.count(), total=len(rows)))
        for feature in rows.select_dtypes(exclude="number"):
            levels = rows[feature].fillna("(missing)").value_counts()
            categorical.extend(dict(years=years, death=label, feature=feature,
                                    level=level, count=count, proportion=count / len(rows))
                               for level, count in levels.items())
    return pd.concat(continuous).rename_axis("feature").reset_index(), pd.DataFrame(categorical)


def binary_metrics(target, probability, prediction):
    return dict(roc_auc=roc_auc_score(target, probability),
                average_precision=average_precision_score(target, probability),
                balanced_accuracy=balanced_accuracy_score(target, prediction),
                brier_score=brier_score_loss(target, probability))


def evaluate_horizon(cohort, predictors, years, output):
    horizon = years * CONFIG["days_per_year"]
    early_death = cohort.death_event.eq(1) & cohort.days_to_event.le(horizon)
    observed = early_death | cohort.days_to_event.ge(horizon)
    X = cohort.loc[observed, predictors].replace([np.inf, -np.inf], np.nan)
    y = early_death.loc[observed].astype(int)
    duration = cohort.loc[observed, "days_to_event"]
    bins = pd.cut(duration, [-np.inf, *CONFIG["duration_bins_days"], np.inf], labels=False)
    strata = y.astype(str) + ":" + bins.astype(str)
    if strata.value_counts().min() < CONFIG["folds"]:
        raise ValueError(f"Insufficient observations in a {years}-year duration/label stratum.")
    descriptive_num, descriptive_cat = gather_descriptives(X, y, years)
    descriptive_num.to_csv(output / "continuous_by_class.csv", index=False)
    descriptive_cat.to_csv(output / "categorical_by_class.csv", index=False)
    model = Pipeline([
        ("preprocessor", clone(median_imputer)),
        ("classifier", RandomForestClassifier(random_state=CONFIG["seed"],
                                              **CONFIG["random_forest"])),
    ])
    splitter = StratifiedKFold(CONFIG["folds"], shuffle=True, random_state=CONFIG["seed"])
    probability = np.full(len(y), np.nan)
    prediction = np.full(len(y), -1)
    fold_numbers = np.zeros(len(y), dtype=int)
    gather_folds = []
    for fold, (train, test) in enumerate(splitter.split(X, strata), 1):
        print(f"{years}-year Random Forest, fold {fold}/{CONFIG['folds']}", flush=True)
        model.fit(X.iloc[train], y.iloc[train])
        probability[test] = model.predict_proba(X.iloc[test])[:, 1]
        prediction[test] = model.predict(X.iloc[test])
        fold_numbers[test] = fold
        gather_folds.append(dict(fold=fold, n_train=len(train), n_test=len(test),
                                 n_deaths=int(y.iloc[test].sum()),
                                 **binary_metrics(y.iloc[test], probability[test], prediction[test])))
    pd.DataFrame(gather_folds).to_csv(output / "fold_metrics.csv", index=False)
    pd.crosstab(fold_numbers, strata).rename_axis("fold").to_csv(output / "fold_strata.csv")
    if len(y) < CONFIG["oof_save_below"]:
        cohort.loc[observed, ["patid", "days_to_event"]].assign(
            death_by_horizon=y, fold=fold_numbers, predicted_death=prediction,
            death_probability=probability,
        ).to_csv(output / "oof_predictions.csv", index=False)

    print(f"{years}-year Random Forest, full-cohort importance fit", flush=True)
    model.fit(X, y)
    preprocessor = model["preprocessor"]
    classifier = model["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    pd.DataFrame({"transformed_feature": feature_names,
                  "importance": classifier.feature_importances_}).sort_values(
                      "importance", ascending=False).to_csv(
                          output / "encoded_feature_importance.csv", index=False)
    feature_map = {f"{transformer}__{name}": name
                   for transformer, _, columns in preprocessor.transformers_ if transformer != "remainder"
                   for name in columns}
    sorted_names = sorted(feature_map, key=len, reverse=True)
    source_names = [next(feature_map[name] for name in sorted_names
                         if encoded == name or encoded.startswith(name + "_"))
                    for encoded in feature_names]
    pd.Series(classifier.feature_importances_, index=source_names).groupby(level=0).sum().reindex(
        predictors, fill_value=0,
    ).sort_values(ascending=False).rename_axis("feature").rename("importance").to_csv(
        output / "feature_importance.csv"
    )
    return dict(years=years, days=horizon, n=len(y), deaths=int(y.sum()),
                censored_early=int((~observed).sum()), death_fraction=y.mean(),
                **binary_metrics(y, probability, prediction))


def cox_design(cohort):
    data = cohort[["HF_type", "Phy_Sex", "Phy_Age", "death_event", "days_to_event"]].copy()
    data["Phy_Age"] = data.Phy_Age.replace([np.inf, -np.inf], np.nan)
    data = data.dropna().reset_index(drop=True)
    group = data.HF_type + " / " + data.Phy_Sex
    levels = sorted(group.unique())
    # Joint indicators allow the HF type association to differ by sex.
    design = pd.DataFrame({level: group.eq(level).astype(float) for level in levels[1:]})
    design[["duration", "death"]] = data[["days_to_event", "death_event"]]
    profiles = pd.DataFrame({level: [float(level == current) for current in levels]
                             for level in levels[1:]}, index=levels)
    return data, group, design, profiles


def compare_cox_groups(cohort, output):
    data, group, design, profiles = cox_design(cohort)
    age = (data.Phy_Age - data.Phy_Age.median()) / 10
    design["age_per_10_years"] = age
    model = lifelines.CoxPHFitter().fit(design, duration_col="duration", event_col="death")
    model.summary.to_csv(output / "cox_coefficients.csv")
    proportional_hazard_test(model, design, time_transform="rank").summary.to_csv(
        output / "cox_ph_tests.csv"
    )
    gather_risks = []
    profiles["age_per_10_years"] = 0.0
    gather_pairwise_hazards(model, profiles).to_csv(
        output / "cox_pairwise_hazard_ratios.csv", index=False
    )
    for level, profile in profiles.iterrows():
        population = pd.DataFrame(dict(profile), index=data.index)
        population["age_per_10_years"] = age
        hazard = model.predict_partial_hazard(population).to_numpy()
        baseline = model.baseline_cumulative_hazard_.iloc[:, 0]
        for years in CONFIG["years"]:
            day = years * CONFIG["days_per_year"]
            cumulative = baseline.asof(day)
            risk = -np.expm1(-cumulative * hazard).mean()
            gather_risks.append(dict(group=level, years=years, age_standardized_mortality=risk,
                                     at_risk=int((group.eq(level) & design.duration.ge(day)).sum())))
    pd.DataFrame(gather_risks).to_csv(output / "cox_age_standardized_mortality.csv", index=False)
    design.groupby(group.rename("group")).agg(patients=("death", "size"), deaths=("death", "sum")).to_csv(
        output / "cox_cohort.csv"
    )
    return dict(cox_patients=len(data), cox_excluded_incomplete=len(cohort) - len(data),
                cox_reference=profiles.index[0], cox_concordance=model.concordance_index_)


def gather_pairwise_hazards(model, profiles):
    gather_contrasts = []
    covariates = model.params_.index
    for reference, group_name in combinations(profiles.index, 2):
        contrast = profiles.loc[group_name, covariates] - profiles.loc[reference, covariates]
        estimate = float(contrast @ model.params_)
        standard_error = float(np.sqrt(contrast @ model.variance_matrix_ @ contrast))
        lower, upper = estimate + norm.ppf([0.025, 0.975]) * standard_error
        gather_contrasts.append(dict(group=group_name, reference=reference,
                                     hazard_ratio=np.exp(estimate), lower_95=np.exp(lower),
                                     upper_95=np.exp(upper),
                                     p_value=2 * norm.sf(abs(estimate / standard_error))))
    return pd.DataFrame(gather_contrasts)


def compare_within_age_strata(cohort, output):
    """Let each completed year of age have its own baseline hazard."""
    data, _, design, profiles = cox_design(cohort)
    design["age_year"] = np.floor(data.Phy_Age).astype(int)
    model = lifelines.CoxPHFitter(strata=["age_year"]).fit(
        design, duration_col="duration", event_col="death"
    )
    gather_pairwise_hazards(model, profiles).to_csv(
        output / "cox_age_stratified_pairwise_hazard_ratios.csv", index=False
    )


def main():
    cohort, predictors = prepare()
    if not cohort.death_event.isin([0, 1]).all() or not (
        np.isfinite(cohort.days_to_event) & cohort.days_to_event.ge(0)
    ).all():
        raise ValueError("Mortality status must be binary with finite nonnegative follow-up.")
    output = PROJECT / CONFIG["output"]
    target_container = CONFIG["years"]
    gather_results = []
    for years in target_container:
        destination = output / f"{years}_year"
        destination.mkdir(exist_ok=True)
        gather_results.append(evaluate_horizon(cohort, predictors, years, destination))
        pd.DataFrame(gather_results).to_csv(output / "classification_summary.csv", index=False)
    print("Fitting age-adjusted Cox proportional hazards model", flush=True)
    cox = compare_cox_groups(cohort, output)
    print("Fitting Cox sensitivity model stratified by completed year of age", flush=True)
    compare_within_age_strata(cohort, output)
    (output / "run_summary.json").write_text(
        json.dumps(dict(**cox, pandas=pd.__version__, sklearn=sklearn.__version__,
                        lifelines=lifelines.__version__), indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
