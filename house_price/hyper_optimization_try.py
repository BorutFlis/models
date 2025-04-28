import os
from functools import partial

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone
from sklearn.preprocessing import FunctionTransformer

DATA_DIR = "../data"

# Define the parameter grids for RandomForest and XGBoost
rf_param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [10, 20, 30, None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

xgb_param_grid = {
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.01, 0.1, 0.3],
    'model__n_estimators': [100, 200, 300],
    'model__min_child_weight': [1, 3, 5],
    'model__subsample': [0.8, 0.9, 1.0]
}





imputers = {
    "SimpleImputer": ColumnTransformer([
        ('num_imputer', SimpleImputer(strategy='median'), 'numeric'),
        ('cat_imputer', SimpleImputer(strategy='most_frequent'), 'categorical')
    ]),
    "KNNImputer": KNNImputer(),
    "IterativeImputer": IterativeImputer()
}

# Define model pipelines
models = {
    "RandomForest": (RandomForestClassifier(random_state=42), rf_param_grid),
    "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), xgb_param_grid)
}


# Function to preprocess data
def preprocess_data(df):
    # Identify categorical and numerical columns
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = df.select_dtypes(include=['number']).columns.tolist()
    numerical_features.remove("SalePrice")

    # One-hot encode categorical features
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
    df_encoded = pd.get_dummies(df, columns=categorical_features)

    return df_encoded, numerical_features


# Function to evaluate model performance
def evaluate_model(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred)
    }


def run_gridCV(
        X,
        y,
        preprocessor: ColumnTransformer,
        model,
        param_grid,
        cv_strategy=None
):
    """
    runs GridSearchCV on model, imputer combination
    it creates the pipeline(imputer, scaler, model) and collect all results

    :return: results of all parameters
    """
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    f1_score_Y = partial(f1_score, pos_label="Y")
    # Perform Grid Search
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring="r2",
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X, y)
    return grid_search


# Main hyperparameter optimization process
def run_hyperparameter_optimization(df):
    results = []

    # Preprocess data
    df, numeric_features = preprocess_data(df)
    X = df.drop(columns=['LABEL_HFD_next'])
    y = df['LABEL_HFD_next']
    groups = df['groups']

    # Define cross-validation strategy
    group_kfold = GroupKFold(n_splits=5)

    # Loop over imputation methods and models
    for imputer_name, imputer in imputers.items():
        print(f"Running for imputer: {imputer_name}")

        # Apply the imputer
        if isinstance(imputer, ColumnTransformer):
            preprocessor = imputer
        else:
            preprocessor = ColumnTransformer([
                ('imputer', imputer, numeric_features)
            ])

        for model_name, (model, param_grid) in models.items():
            print(f"Running model: {model_name} with {imputer_name}")

            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('scaler', StandardScaler()),
                ('model', model)
            ])

            # Perform Grid Search
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=group_kfold.split(X, y, groups=groups),
                scoring='f1',
                n_jobs=-1,
                verbose=2
            )

            grid_search.fit(X, y)

            # Evaluate the best model
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X)
            scores = evaluate_model(y, y_pred)

            # Store results
            results.append({
                'imputer': imputer_name,
                'model': model_name,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                **scores
            })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df


# Load your dataset
df = pd.read_csv("house_price_data/train.csv")

# Run the optimization
# results_df = run_hyperparameter_optimization(df)

# test (model, params) combination for quick GridSearch
test_model_params = (
    RandomForestRegressor(random_state=42),
    {'model__min_samples_split': [2, 5], 'model__max_depth': [10, 20]}
)

df, numeric_features = preprocess_data(df)
X = df.drop(columns=['SalePrice', "Id"])
y = df['SalePrice']

grid_results = run_gridCV(X, y, imputers["KNNImputer"], test_model_params[0], test_model_params[1])
results_df = pd.DataFrame(grid_results.cv_results_)
results_df = pd.concat([
    results_df.drop("params", axis=1),
    results_df["params"].apply(pd.Series)
], axis=1)

# there should be at least some variety in scores
assert results_df["mean_test_score"].nunique() > 1
