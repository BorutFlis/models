import sys
import json

import shap
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import KBinsDiscretizer
import xgboost as xgb

# setting path
sys.path.append('..')
from abstract_models.utils import sort_stratified_regression_group_k, get_most_important_feature

# Define parameter grids for both models
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

xgb_param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0]
}

# Load and prepare data
raw_df = pd.read_csv("house_price_data/train.csv")
df = raw_df.copy()
categorical = (df.dtypes.eq("object") & df.nunique().lt(20)).loc[lambda x: x].index
categorical_int = (df.dtypes.eq("int") & df.nunique().lt(10)).loc[lambda x: x].index
df[categorical] = df.loc[:, categorical].astype("category")
df[categorical_int] = df.loc[:, categorical_int].astype("category")

df = df.select_dtypes(include=["int", "float", "category"])
X = df.dropna(axis=1).reset_index(drop=True).drop(["Id", "SalePrice"], axis=1)
y = np.where(df["SalePrice"].gt(200_000), 1, 0)

# Convert categorical columns to numeric
X = pd.concat([
    X.drop(X.dtypes.eq("category").loc[lambda x: x].index, axis=1),
    pd.get_dummies(X.loc[:, X.dtypes.eq("category")]).astype(int)
], axis=1)

# Step 2: Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Step 3: Train a model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 4: Explain the model predictions using SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# Step 5: Summarize feature importance
# For multi-class, average the absolute SHAP values across classes
mean_abs_shap = np.mean([np.abs(class_shap) for class_shap in shap_values], axis=0)
mean_feature_importance = np.mean(mean_abs_shap, axis=1)

# Put into a nice DataFrame
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'mean_abs_shap': mean_feature_importance
}).sort_values(by='mean_abs_shap', ascending=False)

print(feature_importance)

# Step 6: Select features (e.g., top 2 features)
top_features = feature_importance['feature'].head(2).tolist()
print("Selected features:", top_features)