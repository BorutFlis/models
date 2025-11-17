import os

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, KFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.datasets import load_veterans_lung_cancer

from abstract_models.imputation import median_imputer

DATA_DIR = "../data"

gather_all_dfs = []
all_files_path = os.path.join(DATA_DIR, "raw", "all_files")
for f in os.listdir(os.path.join(DATA_DIR, "raw", "all_files")):
    gather_all_dfs.append(
        pd.read_csv(os.path.join(all_files_path, f), index_col=0)
    )

mortality_data = pd.concat(gather_all_dfs)
mortality_data = mortality_data.loc[mortality_data.loc[:, mortality_data.columns.str.startswith("summary_Blo")].count(axis=1).ge(12)]

hfref_df = pd.read_csv(os.path.join(DATA_DIR, "raw", "hfref_confirmed_HF.csv"))
hfref_df = hfref_df.set_index("patid")

hfpef_df = pd.read_csv(os.path.join(DATA_DIR, "raw", "hfpef_confirmed_HF.csv"))
hfpef_df = hfpef_df.set_index("patid")

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

rsf = RandomSurvivalForest(
    n_estimators=100,
    min_samples_split=10,
    min_samples_leaf=15,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
)
rsf.fit(X, y)

y_pred = pd.Series(rsf.predict(X), index=X.index)

quantiles_container = (0.2, 0.4, 0.6, 0.8, 0.99)
percentile_container = quantiles_container
gather_representive = []
for i, percentile_i in enumerate(percentile_container):
    pct_i = y_pred.quantile(percentile_i)  # 25th percentile of x
    row = (y_pred - pct_i).abs().idxmin()
    gather_representive.append(row)

X_test_representive = X.loc[gather_representive]
X_hfpef_test = X.loc[hfpef_df.index.intersection(X.index)]
X_hfref_test = X.loc[hfref_df.index.intersection(X.index)]

hfpef_surv = rsf.predict_survival_function(X_hfpef_test, return_array=True).mean(axis=0)
hfref_surv = rsf.predict_survival_function(X_hfref_test, return_array=True).mean(axis=0)

surv = rsf.predict_survival_function(X_test_representive, return_array=True)

for i, s in enumerate(surv):
    plt.step(rsf.unique_times_[rsf.unique_times_ < 2500], s[rsf.unique_times_ < 2500], where="post", label=f"Q{i + 1}")
plt.ylabel("Survival probability")
plt.xlabel("Time in days")
plt.legend()
plt.grid(True)


plt.step(rsf.unique_times_[rsf.unique_times_ < 2500], hfpef_surv[rsf.unique_times_ < 2500], where="post", label=f"HFPEF")
plt.step(rsf.unique_times_[rsf.unique_times_ < 2500], hfref_surv[rsf.unique_times_ < 2500], where="post", label=f"HFREF")
plt.ylabel("Survival probability")
plt.xlabel("Time in days")
plt.legend()
plt.grid(True)
