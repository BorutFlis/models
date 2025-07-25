import sys
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import KBinsDiscretizer
import xgboost as xgb
from sklearn.pipeline import Pipeline

from abstract_models.utils import sort_stratified_regression_group_k, get_most_important_feature
from abstract_models.imputation import knn_imputer_missing_robust

# Define parameter grids for both models
rf_param_grid = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [10, 20, 30, None],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4]
}

xgb_param_grid = {
    'xgb__max_depth': [3, 5, 7],
    'xgb__learning_rate': [0.01, 0.1, 0.3],
    'xgb__n_estimators': [100, 200, 300],
    'xgb__min_child_weight': [1, 3, 5],
    'xgb__subsample': [0.8, 0.9, 1.0]
}

# Load and prepare data_loader
raw_df = pd.read_csv("data/train.csv")
df = raw_df.copy()
categorical = (df.dtypes.eq("object") & df.nunique().lt(20)).loc[lambda x: x].index
categorical_int = (df.dtypes.eq("int") & df.nunique().lt(10)).loc[lambda x: x].index
df[categorical] = df.loc[:, categorical].astype("category")
df[categorical_int] = df.loc[:, categorical_int].astype("category")

df = df.select_dtypes(include=["int", "float", "category"])
X = df.reset_index(drop=True).drop(["Id", "SalePrice"], axis=1)
y = df["SalePrice"]

# Convert categorical columns to numeric
X = pd.concat([
    X.drop(X.dtypes.eq("category").loc[lambda x: x].index, axis=1),
    pd.get_dummies(X.loc[:, X.dtypes.eq("category")]).astype(int)
], axis=1)

# Initialize models with pipelines
rf_model = Pipeline([
    ('imputer', knn_imputer_missing_robust),
    ('rf', RandomForestRegressor(random_state=42))
])

xgb_model = Pipeline([
    ('imputer', knn_imputer_missing_robust),
    ('xgb', xgb.XGBRegressor(random_state=42))
])

# Perform GridSearchCV for both models
rf_grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=rf_param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2
)

xgb_grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=xgb_param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2
)

# Fit both models
print("Training RandomForest...")
rf_grid_search.fit(X, y)

print("Training XGBoost...")
xgb_grid_search.fit(X, y)

# Get predictions using best models
rf_predictions = rf_grid_search.best_estimator_.predict(X)
xgb_predictions = xgb_grid_search.best_estimator_.predict(X)

# Calculate R² scores
rf_r2 = r2_score(y, rf_predictions)
xgb_r2 = r2_score(y, xgb_predictions)

# Get best parameters and scores
best_params = {
    'random_forest': rf_grid_search.best_params_,
    'xgboost': xgb_grid_search.best_params_
}

best_scores = {
    'random_forest': {
        'rmse': np.sqrt(-rf_grid_search.best_score_),  # Convert MSE to RMSE
        'r2': rf_r2
    },
    'xgboost': {
        'rmse': np.sqrt(-xgb_grid_search.best_score_),  # Convert MSE to RMSE
        'r2': xgb_r2
    }
}

# Save results to JSON file
results = {
    'best_parameters': best_params,
    'scores': best_scores
}

with open('hyperparameter_optimization_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\nBest Parameters:")
print("RandomForest:", best_params['random_forest'])
print("XGBoost:", best_params['xgboost'])

print("\nBest Scores:")
print("RandomForest - RMSE:", best_scores['random_forest']['rmse'])
print("RandomForest - R²:", best_scores['random_forest']['r2'])
print("XGBoost - RMSE:", best_scores['xgboost']['rmse'])
print("XGBoost - R²:", best_scores['xgboost']['r2'])
