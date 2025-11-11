import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

from abstract_models.imputation import median_imputer, median_imputer_missing
from abstract_models.param_grid import rf_param_grid, xgb_param_grid
from abstract_models.experiment_utils import run_imputation_classifier_grid_search, run_imputation_classifier_random_search
from abstract_models.metric_utils import compute_binary_classification_metrics, plot_multiple_roc_curves
from house_price.data_loader.loader import load_data
from house_price.data_loader.source import RealDataSource

DATA_DIR = "../data"

df = load_data(os.path.join(DATA_DIR, "train.csv"))
data_source = RealDataSource(df)
X, y_container = data_source.xy()


# Classifiers
classifiers = {
    "RandomForest": (RandomForestClassifier(), rf_param_grid),
    "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgb_param_grid)
}

imputers = {
    "median": median_imputer,
    "median_missing": median_imputer_missing
}

model_name = "XGBoost"
model = classifiers[model_name][0]
model_grid = classifiers[model_name][1]

cv_split = data_source.get_cv_split_method()

gather_roc_curve_data = {}
gather_results = []

for target_step in y_container:
    target = target_step.name
    y = target_step
    for model_name in classifiers.keys():
        model = classifiers[model_name][0]
        model_grid = classifiers[model_name][1]
        gather_accuracies = []
        for imputer_name, imputer in imputers.items():
            roc_curve_dict = {
                "y_test": [], "y_proba": []
            }
            gather_roc_curve_data[f"{model_name}_{imputer_name}_{target}"] = roc_curve_dict
            print(f"imputer: {imputer_name}")
            for i, (train_index, test_index) in enumerate(cv_split(X, y)):
                print(f"\ti: {i}")
                X_train = X.iloc[train_index]
                y_train = y.iloc[train_index]
                X_test = X.iloc[test_index]
                y_test = y.iloc[test_index]

                inner_data_source = RealDataSource(df.iloc[train_index])
                cv_inner_split = data_source.get_cv_split_method()

                grid_search_results = run_imputation_classifier_random_search(
                    X_train, y_train, imputer, model, model_grid,
                    cv=cv_inner_split(X_train, y_train), n_iter=5, n_jobs=-1
                )
                y_pred = grid_search_results.predict(X_test)

                # test our assumptions of what y_proba will return
                assert grid_search_results.classes_[1] == 1
                y_proba = grid_search_results.predict_proba(X_test)[:, 1]

                roc_curve_dict["y_test"].extend(y_test.tolist())
                roc_curve_dict["y_proba"].extend(y_proba.tolist())

                results_dict = compute_binary_classification_metrics(y_test, y_pred, y_proba)
                results_dict["n_positive"] = y_test.sum()
                results_dict["n_total"] = len(y_test)

                results_dict["imputer"] = imputer_name
                gather_accuracies.append(
                    results_dict
                )

        results_df = pd.DataFrame(gather_accuracies).assign(target=target).assign(model=model_name).assign(imputer=imputer_name)
        gather_results.append(results_df)
        results_df.to_csv(
            os.path.join(DATA_DIR, "results", f"{model_name}_{data_source.dataset_name}_{target}.csv"), index=None
        )

complete_results_df = pd.concat(gather_results, ignore_index=True)
# aggregate the metrics by folds
grouped_df = (
    complete_results_df.groupby(["target", "model", "imputer"])[["AUC", "Accuracy", "Sensitivity (Recall)", "Specificity"]].mean().reset_index()
)

# if you want to compare different targets
best_models_df = grouped_df.groupby(["target"]).apply(lambda x: x.loc[x["AUC"].idxmax()])