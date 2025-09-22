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
from early_diagnosis.data_loader.loader import load_data
from early_diagnosis.data_loader.source import EarlyDiagnosisCPRDSource

DATA_DIR = "../data"

df = load_data(os.path.join(DATA_DIR, "processed", "early_diagnosis_NT.csv"))

attr_selections = json.load(open(os.path.join(DATA_DIR, "expert_attr_selection.json")))

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

model_name = "LightGBM"
model = classifiers[model_name][0]
model_grid = classifiers[model_name][1]
target_container = [
    'Dia_HFD_6M', 'Dia_HFD_12M', 'Dia_HFD_18M'
]
target = "Dia_HFD_12M"
imputer_name = "median_missing"
imputer = imputers[imputer_name]

pipeline = Pipeline(steps=[('preprocessor', imputer), ('classifier', model)])

gather_roc_curve_data = {}
df_step = df.dropna(subset=target)
# remove all that are eventually diagnosed
df_step = df_step.drop(df_step.index[df_step[target].eq(0) & df_step["Dia_HFD_patient"].eq(1)])
df_step[target].value_counts()
n_positive = df_step[target].value_counts()[1]
negative_df = df_step.loc[df_step["Dia_HFD_patient"].eq(0)].sort_values(by="days_in_db", ascending=False).iloc[:n_positive]

balanced_df = pd.concat(
    [
        df_step.loc[df_step[target].eq(1)],
        negative_df
    ], axis=0
)

data_source = EarlyDiagnosisCPRDSource(balanced_df, target=target)

X, y = data_source.xy()
X = X.drop([
    "ID", "date", 'days_to_HFD', 'days_in_observation', "days_in_db", 'Dia_HFD_patient', 'Dia_HFD_event'],
    axis=1)
attrs = list(set(attr_selections["expert"]).intersection(X.columns)) + ['Med_LD_permanent']
X = X.loc[:, attrs].rename(columns={"Med_LD_permanent": "Med_LD"})

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
