import os
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from abstract_models.imputation import median_imputer, median_imputer_missing
from abstract_models.param_grid import rf_param_grid, xgb_param_grid, lgb_param_grid, rf_imbalanced_param_grid, lgb_imbalanced_param_grid
from abstract_models.experiment_utils import run_imputation_classifier_grid_search, run_imputation_classifier_random_search
from abstract_models.metric_utils import compute_binary_classification_metrics, plot_multiple_roc_curves
from disease_progression.data_loader.loader import load_data
from disease_progression.data_loader.source import ClassificationDPDataSource

DATA_DIR = "../data"

df = load_data(os.path.join(DATA_DIR, "processed", "classification.csv"))

# Classifiers
classifiers = {
    "RandomForest": (RandomForestClassifier(), rf_param_grid),
    "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgb_param_grid),
    "LightGBM": (LGBMClassifier(random_state=42), lgb_imbalanced_param_grid)
}

imputers = {
    "median": median_imputer,
    "median_missing": median_imputer_missing
}

model_name = "RandomForest"
model = classifiers[model_name][0]
model_grid = classifiers[model_name][1]
target_container = [
    'death_2_Y', 'death_5_Y', 'death_10_Y'
]
target = "death_5_Y"
target_container.remove(target)

imputer_name = "median_missing"
imputer = imputers[imputer_name]

pipeline = Pipeline(steps=[('preprocessor', imputer), ('classifier', model)])

gather_roc_curve_data = {}
df_step = df.dropna(subset=target)
df_step[target] = df_step[target].astype(int)
df_step = df_step.drop(target_container + ['days_to_event', 'death_patient'], axis=1)

data_source = ClassificationDPDataSource(df_step, target=target)

X, y = data_source.xy()


cv = data_source.get_cv_split_method()
roc_curve_dict = {
    "y_test": [], "y_proba": [], "records": []
}

# gather accuracies across folds
gather_accuracies = []
for i, (train_index, test_index) in enumerate(cv(X, y)):

    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]
    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index]

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    roc_curve_dict["y_test"].extend(y_test.tolist())
    roc_curve_dict["y_proba"].extend(y_proba.tolist())
    roc_curve_dict["records"].extend(y_test.index)

    results_dict = compute_binary_classification_metrics(y_test, y_pred, y_proba)
    results_dict["n_positive"] = y_test.sum()
    results_dict["n_total"] = len(y_test)
    gather_accuracies.append(
        results_dict
    )

results_df = pd.DataFrame(gather_accuracies)
results_path = os.path.join(DATA_DIR, "results", f"{model_name}_{data_source.dataset_name}_{target}")
results_df.to_csv(os.path.join(results_path))



