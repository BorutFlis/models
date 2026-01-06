import os
import json

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import auc, roc_curve
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from imblearn.over_sampling import ADASYN, SMOTEN

from abstract_models.imputation import (
    median_imputer,
    median_imputer_missing,
    ffill_median_imputer
)
from abstract_models.param_grid import (
    rf_param_grid,
    xgb_param_grid,
    lgb_param_grid
)
from abstract_models.metric_utils import compute_binary_classification_metrics, mean_std_metrics_output
from early_diagnosis.data_loader.loader import load_data
from early_diagnosis.data_loader.source import EarlyDiagnosisSource

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

DATA_DIR = "../data"
attr_selections = json.load(
    open(os.path.join(DATA_DIR, "expert_attr_selection.json"))
)

gather_roc_curve_data = {}
gather_confussion_matrix_data = {}
gather_all_results = []

attr_groups_container = ["expert", "MICE", "expert_blood", "secondary"]
target_container = ["Dia_HFD", "Dia_HFREF", "Dia_HFPEF"]

classifiers = {
    "RandomForest": (RandomForestClassifier(), rf_param_grid),
    "XGBoost": (
        XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        xgb_param_grid
    ),
    "LightGBM": (LGBMClassifier(random_state=42), lgb_param_grid),
}

imputers = {
    "median": median_imputer,
    "median_missing": median_imputer_missing,
    "median_ffill": ffill_median_imputer,
}

# ------------------------------------------------------------------
# Debug / test mode
# ------------------------------------------------------------------

test = True
if test:
    attr_groups_container = ["expert"]
    attr_group = "expert"
    target_container = ["Dia_HFD"]
    target = "Dia_HFD"
    classifiers = {
        "LightGBM": classifiers["LightGBM"],
        "RandomForest": classifiers["RandomForest"],
        "XGBoost": classifiers["XGBoost"],
    }

# ------------------------------------------------------------------
# Main experiment loop
# ------------------------------------------------------------------


df = load_data(os.path.join(DATA_DIR, "raw", "train.csv"))
df = df.dropna(subset=[target])

attrs = attr_selections[attr_group]
data_source = EarlyDiagnosisSource(df, target=target)
X, y = data_source.xy()
X = X.loc[:, attrs]

cv = data_source.get_cv_split_method()

# ==========================================================
# 🔁 OUTER LOOP — MODELS
# ==========================================================
for model_name, (model, model_grid) in classifiers.items():

    print(f"\n=== Model: {model_name} ===")

    # ======================================================
    # 🔁 INNER LOOP — vpop_pct
    # ======================================================
    for i_share in range(0, 6):
        vpop_pct = 0.1 * i_share
        print(f"\nvpop_pct = {vpop_pct:.1f}")

        roc_curve_key = f"{model_name}_{attr_group}_{target}_{i_share}"
        gather_roc_curve_data[roc_curve_key] = {
            "y_test": [],
            "y_proba": []
        }

        gather_confussion_matrix_data[roc_curve_key] = {
            "y_test": [],
            "y_pred": []
        }

        pipeline = Pipeline(
            steps=[
                ("preprocessor", imputers["median"]),
                ("classifier", model),
            ]
        )

        fold_results = []

        for i, (train_idx, test_idx) in enumerate(cv(X, y)):
            print(f"\tFold {i}")

            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

            test_center = (
                cv.keywords["groups"].iloc[test_idx].unique()[0]
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

            gather_roc_curve_data[roc_curve_key]["y_test"].extend(
                y_test.tolist()
            )
            gather_roc_curve_data[roc_curve_key]["y_proba"].extend(
                y_proba.tolist()
            )

            gather_confussion_matrix_data[roc_curve_key][
                "y_test"
            ].extend(y_test.tolist())
            gather_confussion_matrix_data[roc_curve_key][
                "y_pred"
            ].extend(y_pred.tolist())

            metrics = compute_binary_classification_metrics(
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

        results_df.to_csv(
            os.path.join(
                DATA_DIR,
                "results",
                f"{model_name}_{data_source.dataset_name}_{attr_group}_{target}_vpop_{i_share}.csv",
            ),
            index=False,
        )

# ------------------------------------------------------------------
# Final aggregation
# ------------------------------------------------------------------

all_results_df = pd.concat(gather_all_results, ignore_index=True)
