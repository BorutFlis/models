import os
import json

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, auc, roc_curve, accuracy_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from imblearn.over_sampling import ADASYN

from abstract_models.imputation import median_imputer, median_imputer_missing, ffill_median_imputer
from abstract_models.param_grid import rf_param_grid, xgb_param_grid, lgb_param_grid
from abstract_models.experiment_utils import run_imputation_classifier_grid_search, run_imputation_classifier_random_search
from abstract_models.metric_utils import compute_binary_classification_metrics
from early_diagnosis.data_loader.loader import load_data
from early_diagnosis.data_loader.source import EarlyDiagnosisSource

DATA_DIR = "../data"
DATA_DUMP_DIR = "../data_dump"

attr_selections = json.load(open(os.path.join(DATA_DIR, "expert_attr_selection.json")))

file_path = os.path.join(DATA_DIR, "raw", "train.csv")

gather_roc_curve_data = {}
gather_confussion_matrix_data = {}
gather_all_results = []

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
    attr_groups_container = ["expert"]
    target_container = ['Dia_MULTI']
    classifiers = {
        "LightGBM": classifiers["LightGBM"]#,
        # "RandomForest": classifiers["RandomForest"],
        # "XGBoost": classifiers["XGBoost"]
    }

vpop_pct = 0.1

for i in range(0, 6):
    vpop_pct = 0.1 * i
    for attr_group in attr_groups_container:
        for target in target_container:
            df = load_data(file_path)

            df = df.dropna(subset=[target])
            #df = df.rename(columns={"Med_LD_permanent": "Med_LD"})

            attrs = list(set(attr_selections[attr_group]).intersection(df.columns))  # + ["Med_Sta"]

            data_source = EarlyDiagnosisSource(df, target=target)
            data_source.group_col = "ID"
            data_source.target_map = {"HFPEF": 2, "HFREF": 1, "N": 0}

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

                imputer_name = "median"
                imputer = imputers["median"]

                pipeline = Pipeline(steps=[('preprocessor', imputer), ('classifier', model)])

                gather_accuracies = []

                for i, (train_index, test_index) in enumerate(cv(X, y)):
                    print(f"\ti: {i}")
                    X_train = X.iloc[train_index]
                    y_train = y.iloc[train_index]

                    class_distribution = y_train.value_counts()

                    # print(f'Test: {cv.keywords["groups"].iloc[test_index].unique()[0]}')
                    print(class_distribution)

                    if vpop_pct > 0:
                        desired_len = int(len(X_train)/(1-vpop_pct))
                        ada = ADASYN(
                            random_state=42,
                            sampling_strategy={
                                2: y_train.eq(2).sum() + (desired_len - len(X_train))/2,
                                1: y_train.eq(1).sum() + (desired_len - len(X_train))/2
                            }
                        )
                        X_resample, y_resample = ada.fit_resample(pipeline["preprocessor"].fit_transform(X_train), y_train)
                    else:
                        X_resample, y_resample = pipeline["preprocessor"].fit_transform(X_train), y_train

                    print(f"resampled distribution {y_resample.value_counts()}")

                    X_test = X.iloc[test_index]
                    y_test = y.iloc[test_index]

                    pipeline['classifier'].fit(X_resample, y_resample)

                    y_pred = pipeline.predict(X_test)

                    # test our assumptions of what y_proba will return
                    # assert pipeline.classes_[1] == 1
                    # y_proba = pipeline.predict_proba(X_test)[:, 1]

                    results_dict = {}
                    n_0_test = y_test.eq(0).sum()
                    n_1_test = y_test.eq(1).sum()
                    n_2_test = y_test.eq(2).sum()

                    assert (n_0_test + n_1_test + n_2_test) == len(y_test)

                    results_dict["accuracy"] = accuracy_score(y_test, y_pred)
                    results_dict["recall_1"] = (y_test.eq(1) & (y_pred == 1)).sum()/y_test.eq(1).sum()
                    results_dict["recall_2"] = (y_test.eq(2) & (y_pred == 2)).sum()/y_test.eq(2).sum()

                    results_dict["specificity"] = (y_test.eq(0) & (y_pred == 0)).sum()/y_test.eq(0).sum()
                    results_dict["n_negative"] = y_test.eq(0).sum()
                    results_dict["n_total"] = len(y_test)
                    results_dict["n_2_resample"] = y_resample.eq(2).sum()
                    results_dict["n_1_resample"] = y_resample.eq(1).sum()
                    results_dict["n_0_resmpale"] = y_resample.eq(0).sum()

                    results_dict["imputer"] = imputer_name
                    gather_accuracies.append(
                        results_dict
                    )

                results_df = (
                    pd.DataFrame(gather_accuracies).assign(Model=model_name)
                    .assign(attr_group=attr_group).assign(target=target).assign(vpop_pct=vpop_pct)
                )
                gather_all_results.append(results_df)
                results_df.to_csv(
                    os.path.join(DATA_DIR, "results", f"{model_name}_{data_source.dataset_name}_{attr_group}_{target}.csv"),
                    index=None
                )

all_results_df = pd.concat(gather_all_results, ignore_index=True)