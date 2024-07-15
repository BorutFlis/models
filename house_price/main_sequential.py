import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import KBinsDiscretizer

# setting path
sys.path.append('..')
from abstract_models.utils import sort_stratified_regression_group_k, get_most_important_feature


target_variable = "SalePrice"

raw_df = pd.read_csv("house_price_data/train.csv")
df = raw_df.copy()
categorical = (df.dtypes.eq("object") & df.nunique().lt(20)).loc[lambda x: x].index
categorical_int = (df.dtypes.eq("int") & df.nunique().lt(10)).loc[lambda x: x].index
df[categorical] = df.loc[:, categorical].astype("category")
df[categorical_int] = df.loc[:, categorical_int].astype("category")

df = df.select_dtypes(include=["int", "float", "category"])
X = df.dropna().reset_index(drop=True).drop("SalePrice", axis=1)
y = df["SalePrice"]

# Convert the target variable into decile bins
est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
y_binned = est.fit_transform(y.values.reshape(-1, 1)).astype(int).ravel()

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

splits = sort_stratified_regression_group_k(df, "SalePrice")

# Initialize the RandomForestRegressor
model = RandomForestRegressor(random_state=42)

# List to store the RMSE for each fold
rmse_scores = []

gather_r2 = []

gather_stratified_rmse = []

for train_index, test_index in splits:
    # Splitting the data into train and test sets
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    gather_thresholds = []
    for i in range(10):
        gather_thresholds.append(np.quantile(y_train, q=0.1 * i))

    # Training the RandomForestRegressor
    model.fit(X_train, y_train)

    # Predicting the target variable for the test set
    y_pred = model.predict(X_test)

    # Calculating the RMSE for the current fold
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_scores.append(rmse)

    # TODO: run classification metrics
    # TODO: include values outside train set intervals
    y_test_classification = np.digitize(y_test, bins=gather_thresholds)
    y_pred_classification = np.digitize(y_pred, bins=gather_thresholds)

    r2 = r2_score(y_test, y_pred)
    gather_r2.append(r2)

# Output the average RMSE across all folds
average_rmse = np.mean(rmse_scores)

average_r2 = np.mean(gather_r2)

print(average_rmse, average_r2)
