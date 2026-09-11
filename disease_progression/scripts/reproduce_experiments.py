import os
from functools import partial
from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold, GroupKFold, StratifiedKFold, cross_val_score

from abstract_models.param_grid import rf_param_grid, xgb_param_grid, lgb_param_grid, rf_imbalanced_param_grid, lgb_imbalanced_param_grid, nn_param_grid, svm_param_grid
from abstract_models.experiment_utils import run_cross_validation, run_cross_validation_reg
from abstract_models.imputation import median_imputer, median_imputer_missing
from abstract_models.metric_utils import compute_binary_classification_metrics_adjusted, mean_std_metrics_output
from abstract_models.survival import cross_validate_coxph

DATA_DIR = "../data"
RESULTS_DIR = os.path.join(DATA_DIR, "results")

experiments_to_run = ["cross_entropy_lambda_death"] #["mortality_high_risk"]

attr_selection = [
    "Sym_DAR__ever", "Sym_DAR__in_last_year", "Sym_DAR__count", "Pat_DM__ever", "Pat_DM__in_last_year", "Pat_DM__count",
    "Pat_Hyp__ever", "Pat_Hyp__in_last_year", "Pat_Hyp__count", "Phy_Sex", "Phy_Age", 'Phy_Sys__median_1.5Y', 'Phy_Sys__last',
    'Phy_Dia__last', 'Phy_Dia__median_1.5Y', 'Blo_Hb__median_1.5Y', 'Blo_Hb__last', 'Blo_Cre__median_1.5Y', 'Blo_Cre__last'
]

# Classifiers
classifiers = {
    "RandomForest": (RandomForestClassifier(), rf_param_grid),
    "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgb_param_grid),
    "LightGBM": (LGBMClassifier(random_state=42), lgb_imbalanced_param_grid),
    "SVM": (SVC(probability=True, random_state=42), svm_param_grid)
}

imputers = {
    "median": median_imputer,
    "median_missing": median_imputer_missing
}

if "cross_entropy_lambda_death" in experiments_to_run:
    TARGET = "death_event"
    EXPOSURE = "days_to_event"

    df = pd.read_csv(os.path.join(DATA_DIR, "processed", "high_risk_HES_big.csv"), index_col=0)

    X = df.loc[:, attr_selection]
    y = df[TARGET].astype(int)
    exposure = df[EXPOSURE].astype(float)

    model = LGBMClassifier(
        objective="cross_entropy_lambda",
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42
    )
    pipeline = Pipeline(steps=[('preprocessor', median_imputer), ('classifier', model)])

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    auc_scores = []
    logloss_scores = []

    for train_idx, test_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        exposure_train = exposure.iloc[train_idx]
        exposure_test = exposure.iloc[test_idx]

        model = clone(pipeline)

        # Exposure enters the objective here
        model.fit(
            X_train,
            y_train,
            classifier__sample_weight=exposure_train
        )

        # cross_entropy_lambda model output
        y_proba = model.predict_proba(X_test)

        auc_scores.append(
            roc_auc_score(y_test, y_proba[:, 1])
        )


    print("AUC:")
    print(np.mean(auc_scores), "+/-", np.std(auc_scores))


if "horizon_based_classifier" in experiments_to_run:
    df = pd.read_csv(os.path.join(DATA_DIR, "processed", "high_risk_HES_big.csv"), index_col=0)
    gather_targets = []
    for horizon_i in (2, 5, 10):
        target = f"{horizon_i}Y"
        df[target] = pd.Series()
        df.loc[df["days_to_event"].le(horizon_i * 365.25) & df["death_event"].eq(1), target] = 1
        df.loc[df["days_to_event"].gt(horizon_i * 365.25) & df["death_event"].eq(1), target] = 0
        df.loc[df["days_to_event"].gt(horizon_i * 365.25) & df["death_event"].eq(0), target] = 0

        assert (df["days_to_event"].le(horizon_i * 365.25) & df["death_event"].eq(0)).sum() == (len(df) - df[target].count())
        gather_targets.append(target)
    target_container = gather_targets

    gather_results = []
    for target in target_container:
        sub_df = df.dropna(subset=target)
        # patid_container = sub_df.index
        # sample = itemgetter(np.random.choice(len(patid_container), 5000, replace=False))
        # sub_df = sub_df.loc[sample(patid_container)]
        X = sub_df.loc[:, attr_selection]
        y = sub_df[target].astype(int)

        imputer_name = "median"
        imputer = imputers[imputer_name]

        model_name = "RandomForest"
        model = classifiers[model_name][0]
        model_grid = classifiers[model_name][1]

        pipeline = Pipeline(steps=[('preprocessor', imputer), ('classifier', model)])

        base_model_process = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=rf_param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=10,
            verbose=1,
            n_iter=15,
        )
        cv_method = KFold(n_splits=5)
        normal_cv_results = run_cross_validation(
            cv_method.split, base_model_process, compute_binary_classification_metrics_adjusted, X, y
        )
        normal_cv_results["target"] = target
        gather_results.append(normal_cv_results)
    pd.concat(gather_results, ignore_index=True).to_csv(os.path.join(RESULTS_DIR, "horizon_based_classifier.csv"), index=None)

if "mortality_high_risk" in experiments_to_run:
    full_df = pd.read_csv(os.path.join(DATA_DIR, "processed", "high_risk_HES_big.csv"))
    target = "high_risk_1000"

    df = full_df.dropna(subset=target)

    future_leak = [
        'post_all_hosp_total_duration', 'post_all_hosp_n',
        'post_emmergency_hosp_total_duration', 'post_emmergency_hosp_n',
        'post_all_hosp_total_duration_in_30_days', 'post_all_hosp_n_in_30_days',
        'post_all_hosp_total_duration_in_60_days',
        'post_all_hosp_n_in_60_days', 'post_all_hosp_total_duration_in_90_days', 'post_all_hosp_n_in_90_days',
        'post_emmergency_hosp_total_duration_in_30_days', 'post_emmergency_hosp_n_in_30_days',
        'post_emmergency_hosp_total_duration_in_60_days',
        'post_emmergency_hosp_n_in_60_days', 'post_emmergency_hosp_total_duration_in_90_days',
        'post_emmergency_hosp_n_in_90_days',
        'post_emmergency_days_to_hosp'
    ]
    irrelevant_cols = [
        "date", 'cprd_ddate', 'regenddate', 'yob',
        'regstartdate'
    ]
    target_container = [
        'days_to_event', 'death_patient', 'death_event',
        'high_risk_1000', 'high_risk_3000', 'high_risk_5000', "high_risk_1000_adjusted_no_hosp"
    ]
    target_container.remove(target)

    to_drop_cols = future_leak + irrelevant_cols + target_container

    df = df.drop(to_drop_cols, axis=1)
    X = df.drop(target, axis=1)
    y = df[target]

    imputer_name = "median"
    imputer = imputers[imputer_name]

    model_name = "RandomForest"
    model = classifiers[model_name][0]
    model_grid = classifiers[model_name][1]

    pipeline = Pipeline(steps=[('preprocessor', imputer), ('classifier', model)])

    base_model_process = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=rf_param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=10,
        verbose=1,
        n_iter=10,
    )
    cv_method = KFold(n_splits=5)
    normal_cv_results = run_cross_validation(
        cv_method.split, base_model_process, compute_binary_classification_metrics_adjusted, X, y
    )

    stratified_cv_method = StratifiedKFold(n_splits=5)
    stratified_cv_results = run_cross_validation(
        stratified_cv_method.split, base_model_process, compute_binary_classification_metrics_adjusted, X, y
    )

if "hospitalization_regression" in experiments_to_run:
    df = pd.read_csv(os.path.join(DATA_DIR, "processed", "high_risk_HES_big.csv"), index_col=0)

    target_container = [
        'post_all_hosp_total_duration', 'post_all_hosp_n',
        'post_emmergency_hosp_total_duration', 'post_emmergency_hosp_n',
        'post_all_hosp_total_duration_in_30_days', 'post_all_hosp_n_in_30_days',
        'post_all_hosp_total_duration_in_60_days',
        'post_all_hosp_n_in_60_days', 'post_all_hosp_total_duration_in_90_days', 'post_all_hosp_n_in_90_days',
        'post_emmergency_hosp_total_duration_in_30_days', 'post_emmergency_hosp_n_in_30_days',
        'post_emmergency_hosp_total_duration_in_60_days',
        'post_emmergency_hosp_n_in_60_days', 'post_emmergency_hosp_total_duration_in_90_days',
        'post_emmergency_hosp_n_in_90_days',
    ]

    gather_results = []

    sub_df = df.dropna(subset='post_all_hosp_n')
    sub_df = sub_df.loc[:, target_container].fillna(0)
    for target in target_container:
        sub_df = df.dropna(subset=target)
        # patid_container = sub_df.index
        # sample = itemgetter(np.random.choice(len(patid_container), 5000, replace=False))
        # sub_df = sub_df.loc[sample(patid_container)]
        X = sub_df.loc[:, attr_selection]
        y = sub_df[target]

        imputer_name = "median"
        imputer = imputers[imputer_name]

        model_name = "RandomForest"
        model = RandomForestRegressor()
        model_grid = classifiers[model_name][1]
        try:
            del model_grid['classifier__class_weight']
        except KeyError:
            pass

        pipeline = Pipeline(steps=[('preprocessor', imputer), ('classifier', model)])

        base_model_process = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=rf_param_grid,
            cv=5,
            scoring='r2',
            n_jobs=10,
            verbose=1,
            n_iter=3,
        )
        cv_method = KFold(n_splits=5)
        normal_cv_results = run_cross_validation_reg(
            cv_method.split, base_model_process, r2_score, X, y
        )
        normal_cv_results["target"] = target
        gather_results.append(normal_cv_results)
    (pd.concat(gather_results, ignore_index=True).rename(columns={0: "r2"})
       .to_csv(os.path.join(RESULTS_DIR, "hospitalization_regression.csv"), index=None)
     )
if "mortality_high_risk_hospitalization_adjustment" in experiments_to_run:
    full_df = pd.read_csv(os.path.join(DATA_DIR, "processed", "high_risk_HES_big.csv"))
    df = full_df.dropna(subset="high_risk_1000_adjusted_no_hosp")

    target = "high_risk_1000_adjusted_no_hosp"

    future_leak = [
        'post_all_hosp_total_duration', 'post_all_hosp_n',
        'post_emmergency_hosp_total_duration', 'post_emmergency_hosp_n',
        'post_all_hosp_total_duration_in_30_days', 'post_all_hosp_n_in_30_days',
        'post_all_hosp_total_duration_in_60_days',
        'post_all_hosp_n_in_60_days', 'post_all_hosp_total_duration_in_90_days', 'post_all_hosp_n_in_90_days',
        'post_emmergency_hosp_total_duration_in_30_days', 'post_emmergency_hosp_n_in_30_days',
        'post_emmergency_hosp_total_duration_in_60_days',
        'post_emmergency_hosp_n_in_60_days', 'post_emmergency_hosp_total_duration_in_90_days',
        'post_emmergency_hosp_n_in_90_days',
        'post_emmergency_days_to_hosp'
    ]
    irrelevant_cols = [
        "date", 'cprd_ddate', 'regenddate', 'yob',
        'regstartdate'
    ]
    target_container = [
        'days_to_event', 'death_patient', 'death_event',
        'high_risk_1000', 'high_risk_3000', 'high_risk_5000', "high_risk_1000_adjusted_no_hosp"
    ]
    target_container.remove(target)

    to_drop_cols = future_leak + irrelevant_cols + target_container

    df = df.drop(to_drop_cols, axis=1)
    X = df.drop(target, axis=1)
    y = df[target]

    imputer_name = "median"
    imputer = imputers[imputer_name]

    model_name = "RandomForest"
    model = classifiers[model_name][0]
    model_grid = classifiers[model_name][1]

    pipeline = Pipeline(steps=[('preprocessor', imputer), ('classifier', model)])

    base_model_process = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=rf_param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=10,
        verbose=1,
        n_iter=30,
    )
    cv_method = KFold(n_splits=5)
    normal_cv_results = run_cross_validation(
        cv_method.split, base_model_process, compute_binary_classification_metrics_adjusted, X, y
    )

    stratified_cv_method = StratifiedKFold(n_splits=5)
    stratified_cv_results = run_cross_validation(
        stratified_cv_method.split, base_model_process, compute_binary_classification_metrics_adjusted, X, y
    )

if "cox_PH_KFold" in experiments_to_run:
    df = pd.read_csv(os.path.join(DATA_DIR, "processed", "high_risk_HES_big.csv"), index_col=0)

    X_imp_df = pd.DataFrame(
        median_imputer.fit_transform(df.loc[:, attr_selection]),
        index=df.index,
        columns=median_imputer.get_feature_names_out()
    )

    predictions, fold_scores = cross_validate_coxph(
        df=pd.concat([X_imp_df, df[["days_to_event", "death_event"]]], axis=1),
        duration_col="days_to_event", event_col="death_event",
        n_splits=5,
        penalizer=0.01
    )

if "attr_selection" in experiments_to_run:
    df = pd.read_csv(os.path.join(DATA_DIR, "processed", "high_risk_HES_big.csv"))
    df = df.dropna(subset="high_risk_1000_adjusted_no_hosp")


    X = df.loc[:, attr_selection]
    y = df["high_risk_1000_adjusted_no_hosp"].astype(int)
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
        n_iter=2,
    )
    k_fold = KFold(n_splits=5, shuffle=True)
    normal_cv_results = run_cross_validation(
        k_fold.split, base_model_process, compute_binary_classification_metrics_adjusted, X, y
    )



