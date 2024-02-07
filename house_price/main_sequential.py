import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import KBinsDiscretizer


target_variable = "SalePrice"

df = pd.read_csv("house_price_data/train.csv")
df = df.select_dtypes(include=["int64", "float64"])
df = df.dropna()
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Convert the target variable into decile bins
est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
y_binned = est.fit_transform(y.values.reshape(-1, 1)).astype(int).ravel()

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the RandomForestRegressor
model = RandomForestRegressor(random_state=42)

# List to store the RMSE for each fold
rmse_scores = []

gather_r2 = []

for train_index, test_index in skf.split(X, y_binned):
    # Splitting the data into train and test sets
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Training the RandomForestRegressor
    model.fit(X_train, y_train)

    # Predicting the target variable for the test set
    y_pred = model.predict(X_test)

    # Calculating the RMSE for the current fold
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_scores.append(rmse)

    r2 = r2_score(y_test, y_pred)
    gather_r2.append(r2)

# Output the average RMSE across all folds
average_rmse = np.mean(rmse_scores)

average_r2 = np.mean(gather_r2)

print(average_rmse, average_r2)
