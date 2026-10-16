import os
import re

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from abstract_models.survival import chunk_cindex_calculations, cross_validate_coxph
from abstract_models.imputation import median_imputer
from abstract_models.utils import std_mad_ratio


DATA_DIR = "../data"
RESULTS_DIR = os.path.join(DATA_DIR, "results")

DATA_DUMP_DIR = "../data_dump"
df = pd.read_csv(os.path.join(DATA_DIR, "processed", "high_risk_HES_big.csv"))
df["date"] = pd.to_datetime(df["date"])
df["regstartdate"] = pd.to_datetime(df["regstartdate"])

def year_stratification_labels(df: pd.DataFrame, years=(2, 5, 10)) -> pd.DataFrame:
    for year_i in years:
        censored_filter = df["death_patient"].eq(0) & df["days_to_event"].gt(year_i * 365.25)
        df.loc[censored_filter, f"death_{year_i}_Y"] = 0
        dead_filter = df["death_patient"].eq(1) & df["days_to_event"].le(year_i * 365.25)
        df.loc[dead_filter, f"death_{year_i}_Y"] = 1
        alive_until_threshold_filter = df["death_patient"].eq(1) & df["days_to_event"].gt(
            year_i * 365.25)
        df.loc[alive_until_threshold_filter, f"death_{year_i}_Y"] = 0
        assert pd.concat([censored_filter, dead_filter, alive_until_threshold_filter], axis=1).sum(axis=1).lt(2).all()
        remaining_df = df.loc[df[f"death_{year_i}_Y"].isna()]
        assert remaining_df["death_patient"].eq(0).all()
        assert remaining_df["days_to_event"].le(year_i * 365.25).all()
    return df

df = year_stratification_labels(df, years=(1, 2, 5))


schema_df = pd.read_csv(os.path.join(DATA_DIR, "processed", "high_risk_HES_big_schema.csv"))

attr_selection = schema_df.loc[schema_df["variable_type"].eq("predictor"), "attr"].to_list()

results_df = pd.read_csv(os.path.join(RESULTS_DIR, "wei_type_classification.csv"))

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

have_hosp_stats_filter = df.loc[:, hosp_cols].notna().any(axis=1)
df = df.loc[have_hosp_stats_filter]
print(f"after hosp filter {df.shape}")

method_study_df = df.loc[(df["date"] - df["regstartdate"]).dt.days.gt(365)]
print(f"after 1-year registration minimum {df.shape}")

# filtering for the clinical study
hf_type_df = df.dropna(subset="HF_type").reset_index(drop=True)
print(f"after including only HF-type {hf_type_df.shape}")

df = hf_type_df
# pd.concat(
#     [
#         df.groupby("Phy_Sex").size().to_frame("Number"),
#         df.groupby("Phy_Sex")["Phy_Age"].describe().loc[:, ["mean", "std"]].round(3).add_prefix("age_")
#     ], axis=1).to_csv(os.path.join(DATA_DUMP_DIR, "high_risk_HES_big_descriptive.csv"))

clear_examples_weighted_pred_df = pd.merge(
    pd.read_csv(os.path.join(DATA_DIR, "interim", "clear_examples_weighted_preds.csv")),
    df.loc[:, ["patid", "death_event", "days_to_event"]],
    on="patid"
)
high_risk_weighted_pred_df = pd.merge(
    pd.read_csv(os.path.join(DATA_DIR, "interim", "high_risk_weighted_preds.csv")),
    df.loc[:, ["patid", "death_event", "days_to_event"]],
    on="patid"
)
low_risk_weighted_pred_df = pd.merge(
    pd.read_csv(os.path.join(DATA_DIR, "interim", "low_risk_weighted_preds.csv")),
    df.loc[:, ["patid", "death_event", "days_to_event"]],
    on = "patid"
)
normal_pred_df = pd.merge(
    pd.read_csv(os.path.join(DATA_DIR, "interim", "normal_preds.csv")),
    df.loc[:, ["patid", "death_event", "days_to_event"]],
    on="patid"
)
weighted_preds_container = {
    "clear_examples": clear_examples_weighted_pred_df,
    "high_risk": high_risk_weighted_pred_df,
    "low_risk": low_risk_weighted_pred_df,
    "normal": normal_pred_df
}

# gather_cindex = []
# for wei_type, wei_pred_df in weighted_preds_container.items():
#     for target in [ 'death_1_Y', 'death_2_Y', 'death_5_Y']:
#
#         try:
#             concordant, discordant, t1, t2 = chunk_cindex_calculations(
#                 wei_pred_df.loc[:, ["patid", "date", target, "days_to_event", "death_event"]].dropna(subset=target).rename(columns={target: "estimate"}),
#                 chunk=2000,
#                 filter_func=None
#             )
#             cindex_i = concordant/(concordant + discordant)
#         except AssertionError:
#             print(f"{wei_type}, {target} problem with counting")
#             continue
#         gather_cindex.append(
#             {"wei_type": wei_type, "target": target, "cindex": cindex_i}
#         )
#
# cindex_df = pd.DataFrame(gather_cindex)

pipeline = Pipeline(
    steps=[('preprocessor', median_imputer), ('classifier', RandomForestClassifier())]
)

gather_features = []
for target_i in ['death_1_Y', 'death_2_Y', 'death_5_Y']:
    new_pipeline = clone(pipeline)
    target_filter = df[target_i].notna()
    new_pipeline.fit(df.loc[target_filter, attr_selection], df.loc[target_filter, target_i])
    feature_importance_df = pd.DataFrame({
        "features": new_pipeline["preprocessor"].get_feature_names_out(),
        f"{target_i}_importances": new_pipeline["classifier"].feature_importances_
    })
    gather_features.append(
        feature_importance_df
    )

features_df = pd.merge(
    pd.merge(gather_features[0], gather_features[1], on="features"),
    gather_features[2], on="features"
)

features_df["feature_importance_std"] = features_df.loc[
    :, ['death_1_Y_importances', 'death_2_Y_importances', 'death_5_Y_importances']].std(axis=1)

features_df["attr"] = features_df["features"].str.removesuffix("_Y").str.removeprefix("cat__").str.removeprefix("num__")

std_mad_df = std_mad_ratio(df.loc[:, attr_selection].select_dtypes(include=["int", "float"])).dropna().to_frame("std_mad_ratio").reset_index()
features_df = pd.merge(features_df, std_mad_df, left_on="attr", right_on="index", how="left")
features_df["stratify_hf_attr"] = (
    features_df["attr"].str.removesuffix("__median_1.5Y")
    .str.removesuffix("__last").str.removesuffix("__count")
    .str.removesuffix("__age_first").str.removesuffix("__in_last_year").str.removesuffix("__ever")
)

cox_PH_features_df = features_df.sort_values(by='death_1_Y_importances', ascending=False).drop_duplicates(subset="stratify_hf_attr", keep="first")
cox_PH_features_df = cox_PH_features_df.loc[~cox_PH_features_df["stratify_hf_attr"].str.contains("_hosp_")]
cox_PH_features_df.loc[cox_PH_features_df["std_mad_ratio"].lt(2)]
cross_validate_coxph(df.loc[:, cox_PH_features_df.iloc[:8].loc[:, "attr"].to_list() + ["death_event", "days_to_event"]].dropna(how="any"), duration_col="days_to_event", event_col="death_event")