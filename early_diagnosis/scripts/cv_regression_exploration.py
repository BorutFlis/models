import os
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline

from abstract_models.imputation import median_imputer, median_imputer_missing


DATA_DIR = "../data"

df = pd.read_csv(os.path.join(DATA_DIR, "processed", "early_diagnosis_NT.csv"))


attr_selections = json.load(open(os.path.join(DATA_DIR, "expert_attr_selection.json")))

y = df["Blo_NT"]
groups = df["ID"].values

attrs = list(set(attr_selections["expert"]).intersection(df.columns)) + ['Med_LD_permanent']
attrs = list(set(attrs).union(df.columns[df.columns.str.startswith("Blo_")]))
attrs.remove("Blo_NT")

X = df.loc[:, attrs].rename(columns={"Med_LD_permanent": "Med_LD"})

imputer = median_imputer_missing
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
#model = LGBMRegressor()


pipeline = Pipeline(steps=[('preprocessor', imputer), ('classifier', model)])

# ----------------------------------------------------
# Group K-Fold setup
# ----------------------------------------------------
gkf = GroupKFold(n_splits=5)

fold_rmse, fold_r2 = [], []

for fold_idx, (train_idx, valid_idx) in enumerate(gkf.split(X, y, groups)):
    print(f"\n=== Fold {fold_idx + 1} ===")

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # ----------------------------------------------------
    # Model
    # ----------------------------------------------------

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    preds = pipeline.predict(X_valid)

    # Metrics
    mse = mean_squared_error(y_valid, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_valid, preds)

    fold_rmse.append(rmse)
    fold_r2.append(r2)

    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

# ----------------------------------------------------
# Final overall performance
# ----------------------------------------------------
print("\n==============================")
print("Mean RMSE:", np.mean(fold_rmse))
print("Std  RMSE:", np.std(fold_rmse))
print("Mean R²:  ", np.mean(fold_r2))
print("Std  R²:  ", np.std(fold_r2))
print("==============================")


