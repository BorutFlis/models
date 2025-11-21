import os
import json

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from disease_progression.data_loader.loader import load_data
from disease_progression.data_loader.source import ClassificationDPDataSource
from abstract_models.imputation import median_imputer, median_imputer_missing
from abstract_models.param_grid import rf_param_grid, xgb_param_grid, lgb_param_grid, lgb_imbalanced_param_grid

DATA_DIR = "../data"
DATA_DUMP_DIR = "../data_dump"

df = load_data(os.path.join(DATA_DIR, "processed", "classification.csv"))
top_10_container = json.load(open(os.path.join(DATA_DUMP_DIR, "top_10.json")))


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

df = df.drop(target_container + ['days_to_event', 'death_patient'], axis=1)
df = df.dropna(subset=target)

data_source = ClassificationDPDataSource(df, target=target)

X, y = data_source.xy()

imputer_name = "median_missing"
imputer = imputers[imputer_name]

pipeline = Pipeline(steps=[('preprocessor', imputer), ('classifier', model)])

hfref_df = pd.read_csv(os.path.join(DATA_DIR, "raw", "hfref_confirmed_HF.csv"))
hfref_df = hfref_df.set_index("patid")

hfpef_df = pd.read_csv(os.path.join(DATA_DIR, "raw", "hfpef_confirmed_HF.csv"))
hfpef_df = hfpef_df.set_index("patid")

pipeline.fit(X, y)
