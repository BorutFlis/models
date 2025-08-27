import os
import json

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, auc, roc_curve
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier

from abstract_models.imputation import median_imputer, median_imputer_missing, ffill_median_imputer
from abstract_models.param_grid import rf_param_grid, xgb_param_grid, lgb_param_grid
from abstract_models.experiment_utils import run_imputation_classifier_grid_search, run_imputation_classifier_random_search
from abstract_models.metric_utils import compute_binary_classification_metrics
from early_diagnosis.data_loader.loader import load_data
from early_diagnosis.data_loader.source import EarlyDiagnosisSource

DATA_DIR = "../data"
attr_selections = json.load(open(os.path.join(DATA_DIR, "expert_attr_selection.json")))

gather_roc_curve_data = {}
gather_confussion_matrix_data = {}
attr_groups_container = ["expert", "MICE", "expert_blood", "secondary"]
target_container = ["Dia_HFD", "Dia_HFREF", "Dia_HFPEF"]

# Classifiers
classifiers = {
    "RandomForest": (RandomForestClassifier(), rf_param_grid),
    "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgb_param_grid),
    "LightGBM": (LGBMClassifier(random_state=42), lgb_param_grid)
}

imputers = {
    "median": median_imputer,
    "median_missing": median_imputer_missing,
    "median_ffill": ffill_median_imputer
}


def plot_multiple_roc_curves(model_results_dict, title="ROC Curves for Multiple Models"):
    """
    Plots ROC curves for multiple models.

    Parameters:
    -----------
    model_results_dict : dict
        Dictionary with keys as model names and values as (y_test, y_proba) tuples.
        y_proba should be the predicted probability for the positive class.

    title : str
        Title for the plot.
    """

    plt.figure(figsize=(10, 8))

    for model_name, model_results in model_results_dict.items():
        y_test = model_results["y_test"]
        y_proba = model_results["y_proba"]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")

    # Random guess baseline
    plt.plot([0, 1], [0, 1], 'k--', label='Random')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


test = True
if test:
    attr_groups_container = ["expert", "MICE", "expert_blood"]
    target_container = ["Dia_HFD"]
    classifiers = {
        "XGBoost": classifiers["XGBoost"],
        "LightGBM": classifiers[ "LightGBM"]
    }

for attr_group in attr_groups_container:
    for target in target_container:
        df = load_data(os.path.join(DATA_DIR, "raw", "train.csv"))
        echo_cols = df.columns[df.columns.str.match("^Echo_.+")].difference(["Echo_TP"])
        ecg_cols = df.columns[df.columns.str.match("^ECG_.+")].difference(["ECG_TP"])

        df = df.dropna(subset=[target])

        attrs = attr_selections[attr_group] + ["Med_Sta"]
        if attr_group == "secondary":
            df = df.loc[df.loc[:, echo_cols].count(axis=1).gt(5)]

        data_source = EarlyDiagnosisSource(df, target=target)
        X, y = data_source.xy()
        X = X.loc[:, attrs]

        cv = data_source.get_cv_split_method()
        for model_name in classifiers.keys():
            roc_curve_dict = {
                "y_test": [], "y_proba": []
            }
            gather_roc_curve_data[f"{model_name}_{attr_group}_{target}"] = roc_curve_dict

            confusion_matrix_dict = {
                "y_test": [], "y_pred": []
            }
            gather_confussion_matrix_data[f"{model_name}_{attr_group}_{target}"] = confusion_matrix_dict

            model = classifiers[model_name][0]
            model_grid = classifiers[model_name][1]

            imputer_name = "median_ffill"
            imputer = imputers["median_ffill"]
            gather_accuracies = []

            for i, (train_index, test_index) in enumerate(cv(X, y)):
                print(f"\ti: {i}")
                inner_data_source = EarlyDiagnosisSource(df.iloc[train_index], target=target)
                X_train, y_train = data_source.xy()
                X_train = X_train.loc[:, attrs]

                X_test = X.iloc[test_index]
                y_test = y.iloc[test_index]

                cv_inner = data_source.get_cv_split_method()

                grid_search_results = run_imputation_classifier_random_search(
                    X_train, y_train, imputer, model, model_grid,
                    cv=cv_inner(X_train, y_train), n_iter=5, n_jobs=-1
                )
                y_pred = grid_search_results.predict(X_test)

                # test our assumptions of what y_proba will return
                assert grid_search_results.classes_[1] == 1
                y_proba = grid_search_results.predict_proba(X_test)[:, 1]

                roc_curve_dict["y_test"].extend(y_test.tolist())
                roc_curve_dict["y_proba"].extend(y_proba.tolist())

                confusion_matrix_dict["y_test"].extend(y_test.tolist())
                confusion_matrix_dict["y_pred"].extend(y_pred.tolist())

                results_dict = compute_binary_classification_metrics(y_test, y_pred, y_proba)
                results_dict["n_positive"] = y_test.sum()
                results_dict["n_total"] = len(y_test)

                results_dict["imputer"] = imputer_name
                gather_accuracies.append(
                    results_dict
                )

            results_df = (
                pd.DataFrame(gather_accuracies).assign(Model=model_name)
                .assign(attr_group=attr_group).assign(target=target)
            )
            results_df.to_csv(
                os.path.join(DATA_DIR, "results", f"{model_name}_{data_source.dataset_name}_{attr_group}_{target}.csv"),
                index=None
            )
