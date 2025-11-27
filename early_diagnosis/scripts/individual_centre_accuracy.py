import os
import json

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from abstract_models.imputation import median_imputer
from abstract_models.metric_utils import compute_binary_classification_metrics

DATA_DIR = "../data"
DATA_DUMP_DIR = "../data_dump"

model = RandomForestClassifier()
imputer = median_imputer

pipeline = Pipeline(steps=[('preprocessor', imputer), ('classifier', model)])


def cv_by_centre(X, y, group_col):
    centres = group_col.unique()
    print(f"Detected centres: {centres}\n")

    all_metrics = []

    for centre in centres:
        print("=" * 50)
        print(f"Test fold: Centre = {centre}")
        print("=" * 50)

        X_train = X.loc[group_col.ne(centre)]
        y_train = y.loc[group_col.ne(centre)]

        X_test = X.loc[group_col.eq(centre)]
        y_test = y.loc[group_col.eq(centre)]

        class_distribution = y_train.value_counts()
        minority_class = class_distribution.idxmin()
        n_minority_class = class_distribution[minority_class]

        X_minority = X_train.loc[y_train.eq(minority_class)]
        y_minority = y_train.loc[y_train.eq(minority_class)]

        X_majority = X_train.loc[y_train.ne(minority_class)]
        y_majority = y_train.loc[y_train.ne(minority_class)]

        X_resample = pd.concat(
            [
                X_minority,
                X_majority.iloc[np.random.randint(0, len(X_majority), n_minority_class)]
            ]
        )

        y_resample = pd.concat(
            [
                y_minority,
                y_majority.iloc[np.random.randint(0, len(X_majority), n_minority_class)]
            ]
        )
        X_train = X_resample
        y_train = y_resample

        assert y_resample.value_counts().nunique() == 1

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        assert pipeline.classes_[1] == 1
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = compute_binary_classification_metrics(y_test, y_pred, y_pred_proba)
        metrics["centre"] = centre

        all_metrics.append(metrics)

    # Convert to DataFrame
    results_df = pd.DataFrame(all_metrics)


    return results_df


# -----------------------
# Example usage:
# -----------------------
if __name__ == "__main__":

    DATA_PATH = os.path.join(DATA_DIR, "raw", "train.csv")  # <-- change this
    TARGET = "Dia_HFD"  # Binary classification target
    CENTRE_COL = "centre"  # Centre attribute column
    # ----------------------------------------------------------

    # Load dataset
    df = pd.read_csv(DATA_PATH, index_col=[0, 1])
    df = df.dropna(subset=TARGET)

    attr_selections = json.load(open(os.path.join(DATA_DIR, "expert_attr_selection.json")))
    attrs = list(set(attr_selections["expert"]).intersection(df.columns))

    X = df.loc[:, attrs]
    groups = df["centre"]
    y = df[TARGET].map({"Y": 1, "N": 0})

    results = cv_by_centre(
        X, y, groups
    )

    print("\nFull results table:")
    print(results)
