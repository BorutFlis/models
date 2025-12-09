import os

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, KFold
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.datasets import load_veterans_lung_cancer

from abstract_models.imputation import median_imputer

DATA_DIR = "../data"

gather_all_dfs = []
all_files_path = os.path.join(DATA_DIR, "raw", "all_files")
for f in os.listdir(all_files_path):
    gather_all_dfs.append(
        pd.read_csv(os.path.join(all_files_path, f), index_col=0)
    )

mortality_data = pd.concat(gather_all_dfs)
# mortality_data = pd.read_csv(os.path.join(DATA_DIR, "raw", "surv_LVEF.csv"), index_col=0)
# mortality_data.loc[mortality_data.loc[:, mortality_data.columns.str.startswith("summary_Blo")].count(axis=1).ge(12)]

filter_col = "summary_Blo_Cre"

mortality_data = mortality_data.loc[mortality_data[filter_col].notna()].iloc[:2000]
# mortality_data = mortality_data.loc[mortality_data.loc[:, mortality_data.columns.str.startswith("summary_Blo")].count(axis=1).ge(13)]

print(f"Number of patients: {len(mortality_data)}")

# Load example dataset
data_x, data_y = load_veterans_lung_cancer()

# Convert structured array for convenience
X = mortality_data.drop(['death_patient', "days_to_event"], axis=1)
X = pd.DataFrame(median_imputer.fit_transform(X), index=mortality_data.index)

y_df = mortality_data.loc[:, ['death_patient', "days_to_event"]]
y_df["death_patient"] = y_df["death_patient"].astype(bool)
y_df["days_to_event"] = y_df["days_to_event"].astype(float)

y = y_df.apply(lambda x: (x["death_patient"], x["days_to_event"]), axis=1).to_numpy()
y = y.astype(data_y.dtype)

# Initialize Leave-One-Out cross-validation
loo = LeaveOneOut()
n_samples = X.shape[0]

n_folds = 5
cv_split = KFold(n_splits=n_folds)

# Store predictions and true values
predicted_risks = np.zeros(n_samples)
event_indicators = np.zeros(n_samples, dtype=bool)
event_times = np.zeros(n_samples)

estimators = {
    "no regularization": GradientBoostingSurvivalAnalysis(learning_rate=1.0, max_depth=1, random_state=0),
    "learning rate": GradientBoostingSurvivalAnalysis(learning_rate=0.1, max_depth=1, random_state=0),
    "dropout": GradientBoostingSurvivalAnalysis(learning_rate=1.0, dropout_rate=0.1, max_depth=1, random_state=0),
    "subsample": GradientBoostingSurvivalAnalysis(learning_rate=1.0, subsample=0.5, max_depth=1, random_state=0),
}

for name, est in estimators.items():
    # Loop through each leave-one-out split
    for i, (train_index, test_index) in enumerate(cv_split.split(X)):
        print(f"{i * len(test_index)}/{len(X)}")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        est.set_params(n_estimators=100)
        est.fit(X_train, y_train)
        # # Train the Random Survival Forest
        # rsf = RandomSurvivalForest(
        #     n_estimators=100,
        #     min_samples_split=10,
        #     min_samples_leaf=15,
        #     max_features="sqrt",
        #     n_jobs=-1,
        #     random_state=42
        # )
        # rsf.fit(X_train, y_train)

        # Predict risk score (higher = greater risk)
        risk_score = est.predict(X_test)
        predicted_risks[test_index] = risk_score
        event_indicators[test_index] = y_test["Status"]
        event_times[test_index] = y_test["Survival_in_days"]

    # Evaluate performance (C-index)
    cindex = concordance_index_censored(
        event_indicators, event_times, predicted_risks
    )[0]
    print(f"{name} C-index: {cindex:.3f}")

