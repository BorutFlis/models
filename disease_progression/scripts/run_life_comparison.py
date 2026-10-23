import os

import numpy as np
import pandas as pd

DATA_DIR = "../data"
RESULTS_DIR = os.path.join(DATA_DIR, "results")


hosp_cols =  [
    'pre_all_hosp_total_duration', 'pre_all_hosp_n',
    'post_all_hosp_total_duration', 'post_all_hosp_n',
    'pre_emmergency_hosp_total_duration', 'pre_emmergency_hosp_n',
    'post_emmergency_hosp_total_duration', 'post_emmergency_hosp_n',
    'pre_all_hosp_total_duration_in_30_days', 'pre_all_hosp_n_in_30_days',
    'pre_all_hosp_total_duration_in_60_days', 'pre_all_hosp_n_in_60_days',
    'pre_all_hosp_total_duration_in_90_days', 'pre_all_hosp_n_in_90_days',
    'post_all_hosp_total_duration_in_30_days', 'post_all_hosp_n_in_30_days',
    'post_all_hosp_total_duration_in_60_days', 'post_all_hosp_n_in_60_days',
    'post_all_hosp_total_duration_in_90_days', 'post_all_hosp_n_in_90_days',
    'pre_emmergency_hosp_total_duration_in_30_days',
    'pre_emmergency_hosp_n_in_30_days',
    'pre_emmergency_hosp_total_duration_in_60_days',
    'pre_emmergency_hosp_n_in_60_days',
    'pre_emmergency_hosp_total_duration_in_90_days',
    'pre_emmergency_hosp_n_in_90_days',
    'post_emmergency_hosp_total_duration_in_30_days',
    'post_emmergency_hosp_n_in_30_days',
    'post_emmergency_hosp_total_duration_in_60_days',
    'post_emmergency_hosp_n_in_60_days',
    'post_emmergency_hosp_total_duration_in_90_days',
    'post_emmergency_hosp_n_in_90_days', 'post_emmergency_days_to_hosp'
]

df = pd.read_csv(os.path.join(DATA_DIR, "processed", "high_risk_HES_big.csv"), index_col=0)

have_hosp_stats_filter = df.loc[:, hosp_cols].notna().any(axis=1)
df = df.loc[have_hosp_stats_filter]


life_mapping_df = pd.read_csv(os.path.join(DATA_DIR, "processed", "LIFE_Preserved_high_risk_HES_mapping.csv"))

ischaemic_heart_disease_cols =  life_mapping_df.loc[life_mapping_df["mapping_status"].eq("Derived proxy"), "high_risk_HES_schema_attribute"].squeeze().split("|")
ischaemic_heart_disease_cols = list(map(str.strip, ischaemic_heart_disease_cols))
df["Ischemic heart disease"] = np.where(
    df.loc[:, ischaemic_heart_disease_cols].eq("Y").any(axis=1), "Y", "N"
)
schema_df = pd.read_csv(os.path.join(DATA_DIR, "processed", "high_risk_HES_big_schema.csv"))


overlap_attrs = life_mapping_df.loc[life_mapping_df["mapping_status"].isin(["Exact field", "Matched"]), 'high_risk_HES_schema_attribute'].to_list()
non_na_attrs = life_mapping_df.loc[life_mapping_df["mapping_status"].isin(["Exact field", "Matched"]) & life_mapping_df["schema_dtype"].eq("float64"), 'high_risk_HES_schema_attribute'].to_list()
overlap_attrs.append("Ischemic heart disease")

X = df.dropna(subset=non_na_attrs).loc[:, overlap_attrs]
X = pd.concat([X, df.loc[X.index, "pre_emmergency_hosp_n_in_90_days"].gt(1)], axis=1)
X_hfpef = df.loc[X.index, "HF_type"].eq("hfpef")


# use Selenium and beautiful soup to send data to https://tessa-r.shinyapps.io/shiny/ and extract the risk scores
# compare risk scores of LIFE model with OOF predictions
