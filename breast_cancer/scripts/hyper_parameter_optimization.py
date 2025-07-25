import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

from abstract_models.imputation import median_imputer, median_imputer_missing
from abstract_models.param_grid import rf_param_grid, xgb_param_grid
from abstract_models.experiment_utils import run_imputation_classifier_grid_search, run_imputation_classifier_random_search
from abstract_models.metric_utils import compute_binary_classification_metrics

DATA_DIR = "../data"

df = pd.read_csv(os.path.join(DATA_DIR, 'raw/data.csv'))
df = df.reset_index()

n_folds = 5

# Placeholder for group column and target

target_column = "diagnosis"
to_drop_columns = ["id"]

X = df.drop([target_column] + to_drop_columns, axis=1)
y = df[target_column].map({'M': 1, 'B': 0})


# Classifiers
classifiers = {
    "RandomForest": (RandomForestClassifier(), rf_param_grid),
    "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgb_param_grid)
}

imputers = {
    "median": median_imputer
}

model_name = "XGBoost"
model = classifiers[model_name][0]
model_grid = classifiers[model_name][1]

cv = KFold(n_splits=n_folds)

gather_accuracies = []
for imputer_name, imputer in imputers.items():
    print(f"imputer: {imputer_name}")
    for i, (train_index, test_index) in enumerate(cv.split(X, y)):
        print(f"\ti: {i}")
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]

        cv_inner = KFold(n_splits=n_folds)

        grid_search_results = run_imputation_classifier_random_search(
            X_train, y_train, imputer, model, model_grid,
            cv=cv_inner.split(X_train, y_train), n_iter=5, n_jobs=-1
        )
        y_pred = grid_search_results.predict(X_test)

        # test our assumptions of what y_proba will return
        assert grid_search_results.classes_[1] == 1
        y_proba = grid_search_results.predict_proba(X_test)[:, 1]
        results_dict = compute_binary_classification_metrics(y_test, y_pred, y_proba)
        results_dict["n_positive"] = y_test.sum()
        results_dict["n_total"] = len(y_test)

        results_dict["imputer"] = imputer_name
        gather_accuracies.append(
            results_dict
        )

results_df = pd.DataFrame(gather_accuracies)
results_df.to_csv(
    os.path.join(DATA_DIR, "results", f"{model_name}_breast_cancer.csv")
)
