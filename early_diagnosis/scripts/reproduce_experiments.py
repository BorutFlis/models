import os
import json
from functools import partial
from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, GroupKFold
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTEN

from abstract_models.param_grid import rf_param_grid, xgb_param_grid
from abstract_models.experiment_utils import run_walk_forward_validation, run_cross_validation
from abstract_models.imputation import median_imputer
from abstract_models.metric_utils import compute_binary_classification_metrics, compute_binary_classification_metrics_adjusted, mean_std_metrics_output
from early_diagnosis.data_loader.loader import load_data


DATA_DIR = "../data"
RESULTS_DIR = os.path.join(DATA_DIR, "results")
DATA_DUMP_DIR = "../data_dump"

experiments_to_run = ["per_gender_cv"] # ["risk_stratification"]
attr_selections = json.load(open(os.path.join(DATA_DIR, "expert_attr_selection.json")))

if "cv_by_center_class_balance_exploration" in experiments_to_run:
    attr_selections = json.load(
        open(os.path.join(DATA_DIR, "expert_attr_selection.json"))
    )

    attr_group = "expert"
    attrs = attr_selections[attr_group]

    target = "Dia_HFD"

    # can add more classifiers
    classifiers = {
        "RandomForest": (RandomForestClassifier(), rf_param_grid)
    }


    file_location = os.path.join(DATA_DIR, "raw", "train.csv")
    df = pd.read_csv(file_location, index_col=[0, 1])
    df = pd.concat([df.index.to_frame(), df], axis=1)
    df = df.dropna(subset=[target])

    groups = df["centre"]
    n_groups = df["centre"].nunique()
    cv_method = GroupKFold(n_splits=n_groups)
    cv_split_func = partial(cv_method.split, groups=groups)

    X = df.loc[:, attrs]
    y = df[target].map({"Y": 1, "N": 0})
    imputer = median_imputer

    gather_all_results = []

    for model_name, (model, model_grid) in classifiers.items():

        print(f"\n=== Model: {model_name} ===")

        # ======================================================
        # 🔁 INNER LOOP — vpop_pct
        # ======================================================
        for i_share in range(0, 6):
            vpop_pct = 0.1 * i_share
            print(f"\nvpop_pct = {vpop_pct:.1f}")



            pipeline = Pipeline(
                steps=[
                    ("preprocessor", imputer),
                    ("classifier", model),
                ]
            )

            fold_results = []

            for i, (train_idx, test_idx) in enumerate(cv_split_func(X, y)):
                print(f"\tFold {i}")

                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

                test_center = (
                    cv_split_func.keywords["groups"].iloc[test_idx].unique()[0]
                )

                # -------------------------
                # Imputation + Resampling
                # -------------------------
                X_train_imp = pipeline["preprocessor"].fit_transform(
                    X_train
                )

                if vpop_pct > 0:
                    desired_len = int(len(X_train) / (1 - vpop_pct))
                    n_new_samples = desired_len - len(X_train)

                    positive_share = y_train.sum() / len(y_train)

                    new_1_samples = int((1 - positive_share) * n_new_samples)
                    new_0_samples = int(positive_share * n_new_samples)

                    ada = SMOTEN(
                        random_state=42,
                        sampling_strategy={
                            0: new_0_samples + y_train.eq(0).sum(),
                            1: new_1_samples + y_train.eq(1).sum()
                        },
                    )
                    X_res, y_res = ada.fit_resample(
                        X_train_imp, y_train
                    )
                else:
                    X_res, y_res = X_train_imp, y_train

                # -------------------------
                # Train & Predict
                # -------------------------
                pipeline["classifier"].fit(X_res, y_res)
                y_pred = pipeline.predict(X_test)
                y_proba = pipeline.predict_proba(X_test)[:, 1]

                metrics = compute_binary_classification_metrics_adjusted(
                    y_test, y_pred, y_proba
                )

                metrics = pd.concat([
                    metrics,
                    pd.Series(
                        {
                            "n_positive": y_test.sum(),
                            "n_total": len(y_test),
                            "n_positive_train": y_res.sum(),
                            "n_total_train": len(y_res),
                            "imputer": "median",
                            "test_center": test_center,
                            "vpop_pct": vpop_pct,
                        }
                    )
                ])

                fold_results.append(metrics)

            results_df = (
                pd.DataFrame(fold_results)
                .assign(Model=model_name)
                .assign(attr_group=attr_group)
                .assign(target=target)
            )

            gather_all_results.append(results_df)




if "per_age_cohort_evaluation" in experiments_to_run:
    df = load_data(os.path.join(DATA_DIR, "processed", "balanced_ED_NT.csv"))
    df = load_balanced_dataset().dropna(subset=["Phy_Age", "Dia_HFD_12M"])
    X, y = xy_for_expert_model(df)

    base_model_process = make_base_model_process()
    k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    age_prediction_df = collect_cv_predictions(
        k_fold.split,
        base_model_process,
        X,
        y,
        metadata=df.loc[:, ["Phy_Age"]],
    )

    age_prediction_df.to_csv(
        os.path.join(RESULTS_DIR, "per_age_cohort_evaluation_predictions.csv"),
        index=False,
    )

    age_cohort_results_df = binary_metrics_by_age_cohort(
        age_prediction_df["y_true"],
        age_prediction_df["y_proba"],
        age_prediction_df["Phy_Age"],
    )
    age_cohort_results_df.to_csv(
        os.path.join(RESULTS_DIR, "per_age_cohort_evaluation.csv"),
        index=False,
    )

    age_cohort_display_df = age_cohort_results_df.loc[
        :,
        [
            "cohort",
            "n",
            "n_pos",
            "n_neg",
            "prevalence",
            "roc_auc",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
        ],
    ].round(3)
    age_cohort_display_df.to_csv(
        os.path.join(RESULTS_DIR, "per_age_cohort_evaluation_display.csv"),
        index=False,
    )

if "per_gender_cv" in experiments_to_run:
    df = load_data(os.path.join(DATA_DIR, "processed", "balanced_ED_NT.csv"))
    df = df.dropna(subset="pracid")

    attrs = list(set(attr_selections["expert"]).intersection(df.columns)) + ['Med_LD_permanent']
    attrs.remove("Phy_Sex")

    cv_method = GroupKFold(n_splits=2)

    split_func = partial(cv_method.split, groups=df["Phy_Sex"])

    X = df.loc[:, attrs]
    y = df["Dia_HFD_12M"]
    pipeline = Pipeline(
        steps=[('preprocessor', median_imputer), ('classifier', RandomForestClassifier())]
    )

    base_model_process = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=rf_param_grid,
        cv=None,
        scoring='accuracy',
        n_jobs=10,
        verbose=1,
        n_iter=5,
    )
    per_gender_cv_results = run_cross_validation(
        split_func, base_model_process, compute_binary_classification_metrics_adjusted, X, y
    )
    per_gender_cv_results.loc[:, ['Accuracy', 'AUC', 'Precision', 'Sensitivity (Recall)', 'Specificity', 'F1 Score']] = (
        per_gender_cv_results.loc[:, ['Accuracy', 'AUC', 'Precision', 'Sensitivity (Recall)', 'Specificity', 'F1 Score']].round(3)
    )
    per_gender_cv_results.to_csv(os.path.join(DATA_DIR, "results", "per_gender_csv.csv"))

if "per_practice_cv" in experiments_to_run:
    df = load_data(os.path.join(DATA_DIR, "processed", "balanced_ED_NT.csv"))
    df = df.dropna(subset="pracid")
    df["pracid"] = df["pracid"].astype(int)

    attrs = list(set(attr_selections["expert"]).intersection(df.columns)) + ['Med_LD_permanent']

    cv_method = GroupKFold(n_splits=10)

    split_func = partial(cv_method.split, groups=df["pracid"])

    X = df.loc[:, attrs]
    y = df["Dia_HFD_12M"]
    pipeline = Pipeline(
        steps=[('preprocessor', median_imputer), ('classifier', RandomForestClassifier())]
    )

    base_model_process = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=rf_param_grid,
        cv=None,
        scoring='accuracy',
        n_jobs=10,
        verbose=1,
        n_iter=5,
    )
    per_practice_cv_results = run_cross_validation(split_func, base_model_process, compute_binary_classification_metrics_adjusted, X, y)
    k_fold = KFold(n_splits=10, shuffle=True)
    normal_cv_results = run_cross_validation(
        k_fold.split, base_model_process, compute_binary_classification_metrics_adjusted, X, y
    )

    per_practice_cmp_df = pd.concat([
        per_practice_cv_results.assign(Validation_Type="By Practice"),
        normal_cv_results.assign(Validation_Type="K-Fold")
    ])
    per_practice_cmp_df.to_csv(os.path.join(DATA_DIR, "results", "per_practice_cv.csv"))

    per_practice_cv_display_df = mean_std_metrics_output(
        per_practice_cmp_df,
        groupby_col="Validation_Type"
    )
    per_practice_cv_display_df.to_csv(os.path.join(DATA_DIR, "results", "per_practice_cv_display.csv"))

if "sliding_window_validation" in experiments_to_run:
    df = load_data(os.path.join(DATA_DIR, "processed", "balanced_ED_NT.csv"))

    attrs = list(set(attr_selections["expert"]).intersection(df.columns)) + ['Med_LD_permanent']

    X = df.loc[:, attrs]
    y = df["Dia_HFD_12M"]

    year = pd.to_datetime(df["date"]).dt.year
    year_group = pd.cut(year, bins=[year.min(), 2008, 2012, 2016, 2020, year.max()], include_lowest=True)


    pipeline = Pipeline(
        steps=[('preprocessor', median_imputer), ('classifier', RandomForestClassifier())]
    )

    base_model_process = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=rf_param_grid,
        cv=None,
        scoring='accuracy',
        n_jobs=2,
        verbose=1,
        n_iter=30,
    )

    time_cv_results = run_walk_forward_validation(
        X, y, year_group, base_model_process, metric_function=compute_binary_classification_metrics_adjusted
    )
    time_cv_df = pd.DataFrame(time_cv_results)
    time_cv_df.to_csv(os.path.join(DATA_DIR, "results", "sliding_window_validation.csv"))
    pd.concat([
        X.groupby(year_group).size().to_frame("Number of Test Examples"),
        y.groupby(year_group).value_counts().unstack()
    ], axis=1).to_csv(os.path.join(DATA_DIR, "results", "sliding_window_descriptives.csv"))

if "survival_analysis_dp" in experiments_to_run:
    pass
if "risk_stratification" in experiments_to_run:
    full_df = pd.read_csv(os.path.join(DATA_DIR, "processed", "risk_stratification.csv"), index_col=[0, 1])
    target_container = ["2Y", "5Y", "10Y"]
    gather_results = []

    patid_container = full_df.index.get_level_values("ID").unique()

    for target_i in target_container:

        sample = itemgetter(np.random.choice(len(patid_container), 200, replace=False))
        df = full_df.loc[sample(patid_container)]

        df = df.dropna(subset=target_i)
        X = df.drop(target_container, axis=1)
        y = df[target_i].map({"Y": 1, "N": 0})

        pipeline = Pipeline(
            steps=[('preprocessor', median_imputer), ('classifier', RandomForestClassifier())]
        )

        base_model_process = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=rf_param_grid,
            cv=None,
            scoring='accuracy',
            n_jobs=10,
            verbose=1,
            n_iter=5,
        )

        cv_method = GroupKFold(n_splits=5)

        split_func = partial(cv_method.split, groups=df.index.get_level_values("ID"))
        normal_cv_results = run_cross_validation(
            split_func, base_model_process, compute_binary_classification_metrics_adjusted, X, y
        )
        normal_cv_results["target"] = target_i
        gather_results.append(
            normal_cv_results
        )
        results_df = pd.concat(gather_results, ignore_index=True)

        agg_results_df = pd.concat([
            results_df.groupby("target", sort=False)[
                ['Accuracy', 'AUC', 'Precision', 'Sensitivity (Recall)', 'Specificity', 'F1 Score']].mean().round(3),
            results_df.groupby("target", sort=False)[['tn', 'fp', 'tp', 'fn']].sum().astype(int)
        ], axis=1)
        agg_results_df.to_csv(os.path.join(DATA_DIR, "results", "risk_stratification_test.csv"))


