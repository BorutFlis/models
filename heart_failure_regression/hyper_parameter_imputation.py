import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Import imputation methods from abstract_models
from abstract_models.imputation import (
    median_imputer,
    knn_imputer,
    iterative_imputer,
    knn_imputer_missing,
    median_imputer_missing
)

LABEL_column = "HouseClass"
DATA_DIR = "data"

# Parameter grids
rf_param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [10, 20, 30, 50],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

xgb_param_grid = {
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1, 0.3],
    'classifier__n_estimators': [100, 200, 300],
    'classifier__min_child_weight': [1, 3, 5],
    'classifier__subsample': [0.8, 0.9, 1.0]
}

# Define preprocessing strategies using imported imputers
imputation_methods = {
    "median_imputer": median_imputer,
    "knn_imputer": knn_imputer,
    "iterative_imputer": iterative_imputer,
    "knn_imputer_missing": knn_imputer_missing,
    "median_imputer_missing": median_imputer_missing
}

# Define model pipelines
models = {
    "RandomForest": (RandomForestClassifier(random_state=42), rf_param_grid),
    "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), xgb_param_grid)
}
