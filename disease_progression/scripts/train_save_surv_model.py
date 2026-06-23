import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, KFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.datasets import load_veterans_lung_cancer

from abstract_models.imputation import median_imputer

DATA_DIR = "../data"

attrs = [
    'Phy_Age', 'Blo_Hb', 'Blo_Alb', 'Blo_Ure', 'Phy_Wei', 'Blo_Cre',
    'Phy_Dia', 'Med_LD', 'Blo_Sod', 'Phy_Sys', 'Blo_MCV', 'Blo_EGFR',
    'Blo_Pot', 'Phy_BMI', 'Blo_TotC'
]

mortality_data = pd.read_csv(os.path.join(DATA_DIR, "processed", "high_risk_HES.csv"), index_col=0)
mortality_data = mortality_data.iloc[np.random.choice(len(mortality_data), 10000, replace=False)]


mortality_data = mortality_data.rename(columns={'summary_Med_LD_ever_taken': "Med_LD"})

mortality_data.columns = mortality_data.columns.str.lstrip("summary_")

# Load example dataset
data_x, data_y = load_veterans_lung_cancer()

X = mortality_data.loc[:, attrs]

y_df = mortality_data.loc[:, ['death_patient', "days_to_event"]]
y_df["death_patient"] = y_df["death_patient"].astype(bool)
y_df["days_to_event"] = y_df["days_to_event"].astype(float)

y = y_df.apply(lambda x: (x["death_patient"], x["days_to_event"]), axis=1).to_numpy()
y = y.astype(data_y.dtype)

rf_surv = RandomSurvivalForest()
rf_surv.fit(X, y)

with open("../data_dump/rf_surv.pkl", "wb") as f:
    pickle.dump(rf_surv, f)
