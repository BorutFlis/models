import os
import json
import re

import pandas as pd
import numpy as np

DATA_DIR = "../data"

df = pd.read_csv(os.path.join(DATA_DIR, "processed", "early_diagnosis_NT.csv"), index_col=[0, 1])

target = "Dia_HFD_12M"

gather_roc_curve_data = {}
df_step = df.dropna(subset=target)
# remove all that are eventually diagnosed
df_step = df_step.drop(df_step.index[df_step[target].eq(0) & df_step["Dia_HFD_patient"].eq(1)])
df_step[target].value_counts()
n_positive = df_step[target].value_counts()[1]
negative_df = df_step.loc[df_step["Dia_HFD_patient"].eq(0)].sort_values(by="days_in_db", ascending=False).iloc[:n_positive]

balanced_df = pd.concat(
    [
        df_step.loc[df_step[target].eq(1)],
        negative_df
    ], axis=0
)

balanced_train_df = balanced_df.copy()

hf_pef_confirmed_HF_df = pd.read_csv(os.path.join(DATA_DIR, "raw", "hfpef_confirmed_HF.csv"))
hf_pef_at_risk_df = pd.read_csv(os.path.join(DATA_DIR, "raw", "hfpef_at_risk.csv"))
hf_pef_df = pd.concat([
    hf_pef_confirmed_HF_df,
    hf_pef_at_risk_df
], ignore_index=True)
hf_pef_df = hf_pef_df.set_index("patid").add_suffix("_hfpef")
hf_pef_df.index.set_names("ID", inplace=True)

hf_ref_confirmed_HF_df = pd.read_csv(os.path.join(DATA_DIR, "raw", "hfref_confirmed_HF.csv"))
hf_ref_at_risk_df = pd.read_csv(os.path.join(DATA_DIR, "raw", "hfref_at_risk.csv"))
hf_ref_df = pd.concat([
    hf_ref_confirmed_HF_df,
    hf_ref_at_risk_df
], ignore_index=True)
hf_ref_df = hf_ref_df.set_index("patid").add_suffix("_hfref")
hf_ref_df.index.set_names("ID", inplace=True)

balanced_df = pd.merge(balanced_df, hf_pef_df["obsdate_hfpef"], left_index=True, right_index=True, how="left")
balanced_df = pd.merge(balanced_df, hf_ref_df["obsdate_hfref"], left_index=True, right_index=True, how="left")

inconsistencies = balanced_df.loc[balanced_df["obsdate_hfpef"].notna() & balanced_df["obsdate_hfref"].notna()].index
balanced_df = balanced_df.drop(inconsistencies)
balanced_df["Dia_MULTI"] = pd.Series()
balanced_df["Dia_MULTI"] = balanced_df["Dia_MULTI"].mask(balanced_df["obsdate_hfpef"].notna() & balanced_df["Dia_HFD_12M"], "HFPEF")
balanced_df["Dia_MULTI"] = balanced_df["Dia_MULTI"].mask(balanced_df["obsdate_hfref"].notna() & balanced_df["Dia_HFD_12M"], "HFREF")
balanced_df["Dia_MULTI"] = balanced_df["Dia_MULTI"].mask(balanced_df["Dia_HFD_12M"].eq(0), "N")

balanced_df = balanced_df.dropna(subset="Dia_MULTI")
balanced_df.to_csv(os.path.join(DATA_DIR, "processed", "cprd_multiclass.csv"))