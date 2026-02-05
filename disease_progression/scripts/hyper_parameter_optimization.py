import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

from abstract_models.imputation import median_imputer, median_imputer_missing
from abstract_models.param_grid import (
    rf_param_grid, xgb_param_grid, lgb_param_grid, lgb_imbalanced_param_grid, svm_param_grid, nn_param_grid
)
from abstract_models.experiment_utils import run_imputation_classifier_grid_search, run_imputation_classifier_random_search
from abstract_models.metric_utils import compute_binary_classification_metrics, plot_multiple_roc_curves
from abstract_models.pytorch_classifier import PyTorchNeuralNetworkClassifier
from disease_progression.data_loader.loader import load_data
from disease_progression.data_loader.source import ClassificationDPDataSource

DATA_DIR = "../data"

healthy_days_in_db_container = [1000, 3000, 5000]
healthy_days_in_db = healthy_days_in_db_container[1]
target = "high_risk"

df = load_data(os.path.join(DATA_DIR, "processed", "full_hes_mortality.csv"))
df[target] = pd.Series()
df.loc[df["days_to_event"].lt(90) & df["death_patient"].eq(1), target] = 1
df.loc[df["days_to_event"].gt(healthy_days_in_db) & df["death_patient"].eq(0), target] = 0

target_container = [
    'death_2_Y', 'death_5_Y', 'death_10_Y', target
]

target_container.remove(target)

df_step = df.dropna(subset=[target])
df_step[target] = df_step[target].astype(int)
df_step = df_step.drop(['days_to_event', 'death_patient', 'date'] + ['post_hosp_total_duration', 'post_hosp_n'], axis=1)

data_source = ClassificationDPDataSource(df_step, target=target)
data_source.dataset_name = "HES_addition"
X, y = data_source.xy()


# Classifiers
classifiers = {
    #"RandomForest": (RandomForestClassifier(), rf_param_grid),
    #"XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgb_param_grid),
    #"LightGBM": (LGBMClassifier(random_state=42), lgb_imbalanced_param_grid),
    "PyTorchNN": (PyTorchNeuralNetworkClassifier(random_state=42), nn_param_grid),
    "SVM": (SVC(probability=True, random_state=42), svm_param_grid)
}

imputers = {
    "median": median_imputer #,
    #"median_missing": median_imputer_missing
}

# model_name = "XGBoost"
# model = classifiers[model_name][0]
# model_grid = classifiers[model_name][1]

cv_split = data_source.get_cv_split_method()

gather_roc_curve_data = {}
gather_results = []

for model_name in classifiers.keys():
    model = classifiers[model_name][0]
    model_grid = classifiers[model_name][1]
    for imputer_name, imputer in imputers.items():
        gather_accuracies = []
        roc_curve_dict = {
            "y_test": [], "y_proba": []
        }
        gather_roc_curve_data[f"{model_name}_{imputer_name}_{target}"] = roc_curve_dict
        print(f"imputer: {imputer_name}")
        for i, (train_index, test_index) in enumerate(cv_split(X, y)):
            print(f"\ti: {i}")
            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            X_test = X.iloc[test_index]
            y_test = y.iloc[test_index]

            inner_data_source = ClassificationDPDataSource(df_step.iloc[train_index], target=target)
            cv_inner = inner_data_source.get_cv_split_method()

            grid_search_results = run_imputation_classifier_random_search(
                X_train, y_train, imputer, model, model_grid,
                cv=cv_inner(X_train, y_train), n_iter=3, n_jobs=-1
            )
            y_pred = grid_search_results.predict(X_test)

            # test our assumptions of what y_proba will return
            assert grid_search_results.classes_[1] == 1
            y_proba = grid_search_results.predict_proba(X_test)[:, 1]

            roc_curve_dict["y_test"].extend(y_test.tolist())
            roc_curve_dict["y_proba"].extend(y_proba.tolist())

            results_dict = compute_binary_classification_metrics(y_test, y_pred, y_proba)
            results_dict["n_positive"] = y_test.sum()
            results_dict["n_total"] = len(y_test)

            results_dict["imputer"] = imputer_name
            gather_accuracies.append(
                results_dict
            )

        results_df = pd.DataFrame(gather_accuracies).assign(target=target).assign(model=model_name).assign(imputer=imputer_name)
        gather_results.append(results_df)
        results_df.to_csv(
            os.path.join(DATA_DIR, "results", f"{model_name}_{imputer_name}_{data_source.dataset_name}_{target}.csv"), index=None
        )

complete_results_df = pd.concat(gather_results, ignore_index=True)
# aggregate the metrics by folds
grouped_df = (
    complete_results_df.groupby(["target", "model", "imputer"])[["AUC", "Accuracy", "Sensitivity (Recall)", "Specificity"]].mean().reset_index()
)

# if you want to compare different targets
best_models_df = grouped_df.groupby(["target"]).apply(lambda x: x.loc[x["AUC"].idxmax()])