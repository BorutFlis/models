import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, KFold
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.datasets import load_veterans_lung_cancer

from abstract_models.imputation import median_imputer
from build.lib.abstract_models.imputation import median_imputer

DATA_DIR = "../data"

attrs = [
    'Phy_Age', 'Blo_Hb', 'Blo_Alb', 'Blo_Ure', 'Phy_Wei', 'Blo_Cre',
    'Phy_Dia', 'Med_LD', 'Blo_Sod', 'Phy_Sys', 'Blo_MCV', 'Blo_EGFR',
    'Blo_Pot', 'Phy_BMI', 'Blo_TotC'
]

mortality_data = pd.read_csv(os.path.join(DATA_DIR, "processed", "high_risk_HES.csv"), index_col=0)
mortality_data = mortality_data.iloc[np.random.choice(len(mortality_data), 10000, replace=False)]


mortality_data = mortality_data.rename(columns={'summary_Med_LD_ever_taken': "Med_LD"})
mortality_data["Med_LD"] = mortality_data["Med_LD"].map({1: "Y", 0: "N"})

mortality_data.columns = mortality_data.columns.str.lstrip("summary_")

# Load example dataset
data_x, data_y = load_veterans_lung_cancer()

X = median_imputer.fit_transform(mortality_data.loc[:, attrs])
X = pd.DataFrame(X, columns=median_imputer.get_feature_names_out())

y_df = mortality_data.loc[:, ['death_patient', "days_to_event"]]
y_df["death_patient"] = y_df["death_patient"].astype(bool)
y_df["days_to_event"] = y_df["days_to_event"].astype(float)

y = y_df.apply(lambda x: (x["death_patient"], x["days_to_event"]), axis=1).to_numpy()
y = y.astype(data_y.dtype)

surv_model = GradientBoostingSurvivalAnalysis()
surv_model.fit(X, y)

with open("../data_dump/surv.pkl", "wb") as f:
    pickle.dump(surv_model, f)
