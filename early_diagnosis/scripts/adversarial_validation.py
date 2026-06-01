"""
Adversarial validation with Random Forest

Goal:
- Label train rows as 0, test rows as 1
- Fit a classifier to predict the label
- AUC near 0.5 => similar distributions
- Higher AUC => covariate shift / distribution mismatch
"""

from __future__ import annotations
import os

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


DATA_DIR = "../data"

def adversarial_validation_rf(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    n_estimators: int = 500,
    max_depth: int | None = None,
    n_jobs: int = -1,
    min_samples_leaf: int = 1,
) -> dict:
    """
    Returns:
        dict with:
            - auc_oof: overall out-of-fold AUC
            - auc_folds: list of AUC per fold
            - oof_pred: OOF predicted probabilities for label=1 (test)
            - feature_importances: DataFrame sorted desc
            - pipeline: fitted pipeline on full combined dataset (useful for reuse)
    """

    if not isinstance(X_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_train and X_test must be pandas DataFrames.")

    # Align columns (keep shared columns only, in same order)
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    if len(common_cols) == 0:
        raise ValueError("No common columns between X_train and X_test.")
    X_train = X_train[common_cols].copy()
    X_test = X_test[common_cols].copy()

    # Create adversarial dataset
    X_adv = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    y_adv = np.r_[np.zeros(len(X_train), dtype=int), np.ones(len(X_test), dtype=int)]

    # Column types
    cat_cols = X_adv.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X_adv.columns if c not in cat_cols]

    # Preprocessing
    numeric_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, num_cols),
            ("cat", categorical_tf, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        random_state=random_state,
        class_weight="balanced",
    )

    pipe = Pipeline(steps=[("pre", pre), ("rf", rf)])

    # CV with OOF predictions
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pred = np.zeros(len(X_adv), dtype=float)
    auc_folds: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_adv, y_adv), start=1):
        X_tr, X_va = X_adv.iloc[tr_idx], X_adv.iloc[va_idx]
        y_tr, y_va = y_adv[tr_idx], y_adv[va_idx]

        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_va)[:, 1]
        oof_pred[va_idx] = p

        auc = roc_auc_score(y_va, p)
        auc_folds.append(auc)
        print(f"Fold {fold}: AUC = {auc:.4f}")

    auc_oof = roc_auc_score(y_adv, oof_pred)
    print(f"\nOOF AUC (overall): {auc_oof:.4f}")

    # Fit on full data to extract importances + feature names
    pipe.fit(X_adv, y_adv)

    # Get feature names after preprocessing
    feature_names = pipe.named_steps["pre"].get_feature_names_out()
    importances = pipe.named_steps["rf"].feature_importances_

    fi = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "auc_oof": auc_oof,
        "auc_folds": auc_folds,
        "oof_pred": oof_pred,
        "feature_importances": fi,
        "pipeline": pipe,
        "common_columns": common_cols,
        "num_columns": num_cols,
        "cat_columns": cat_cols,
    }


if __name__ == "__main__":
    # Example usage:
    # X_train = pd.read_csv("train_features.csv")
    # X_test  = pd.read_csv("test_features.csv")
    # result = adversarial_validation_rf(X_train, X_test)
    # print(result["feature_importances"].head(30))
    anomaly_df = pd.read_csv(os.path.join(DATA_DIR, "processed/ED_anomaly.csv"), index_col=[0, 1])
    df = pd.read_csv(os.path.join(DATA_DIR, "processed/balanced_ED_NT.csv"), index_col=[0, 1])