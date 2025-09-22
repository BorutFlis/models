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
attrs = list(set(attr_selections["expert"]).intersection(X.columns)) + ['Med_LD']
X = X.rename(columns={"Med_LD_permanent": "Med_LD"}).loc[:, attrs]
X["Med_LD"] = X["Med_LD"].map({1:"Y", 0: "N"})
pipeline.fit(X, y)

retro_df = load_data(os.path.join(DATA_DIR, "raw", "train.csv"))
retro_df = retro_df.dropna(subset=["Blo_NT", "Dia_HFD"], axis=0, how="any")
retro_df = retro_df.rename(columns={"Dia_HFD": target})
y_test = retro_df[target].map({"Y": 1, "N": 0})
X_test = retro_df.loc[:, attrs]


y_pred = pipeline.predict(X_test)

# test our assumptions of what y_proba will return
assert pipeline.classes_[1] == 1
y_proba = pipeline.predict_proba(X_test)[:, 1]