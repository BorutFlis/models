import os

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, KFold
from lightgbm import LGBMClassifier
from sksurv.metrics import concordance_index_censored

from abstract_models.imputation import median_imputer
from abstract_models.param_grid import lgb_param_grid
from abstract_models.experiment_utils import run_cross_validation, run_cross_validation_wei, run_cross_validation_wei_preds
from abstract_models.metric_utils import compute_binary_classification_metrics_adjusted

DATA_DIR = "../data"
DATA_DUMP_DIR = "../data_dump"
RESULTS_DIR = os.path.join(DATA_DIR, "results")

# "zero_days",
targets = ['death_1_Y', 'death_2_Y', 'death_5_Y']

attr_selection = [
    "Sym_DAR__ever", "Sym_DAR__in_last_year", "Sym_DAR__count",
    "Pat_DM__ever", "Pat_DM__in_last_year", "Pat_DM__count",
    "Pat_Hyp__ever", "Pat_Hyp__in_last_year", "Pat_Hyp__count",
    "Phy_Sex", "Phy_Age", "Phy_Sys__median_1.5Y", "Phy_Sys__last",
    "Phy_Dia__last", "Phy_Dia__median_1.5Y", "Blo_Hb__median_1.5Y",
    "Blo_Hb__last", "Blo_Cre__median_1.5Y", "Blo_Cre__last",
]

df = pd.read_csv(os.path.join(DATA_DIR, "processed", "high_risk_HES_big.csv"), index_col=[0, 1])
df = df.rename(columns={"death_event": "zero_days"})

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

df = year_stratification_labels(df, years=(1, 2, 5))
have_hosp_stats_filter = df.loc[:, hosp_cols].notna().any(axis=1)
df = df.loc[have_hosp_stats_filter]

filter_no_immediate_hosp = df['post_emmergency_hosp_n_in_90_days'].isna() | df['post_emmergency_hosp_n_in_90_days'].eq(0)

wei_df = pd.DataFrame(columns=targets, index=df.index)
# wei_df.loc[df["death_1_Y"].eq(1), "death_1_Y"] = (df.loc[df["death_1_Y"].eq(1), "days_to_event"]/365.25).apply(lambda x: - np.log(x)) + 1
wei_df.loc[df["death_1_Y"].notna(), "death_1_Y"] = wei_df.loc[df["death_1_Y"].notna(), "death_1_Y"].fillna(1)
wei_df.loc[df["death_1_Y"].eq(0) & df["zero_days"].eq(0) & filter_no_immediate_hosp, "death_1_Y"] = (df.loc[df["death_1_Y"].eq(0) & df["zero_days"].eq(0), "days_to_event"]/5_000).apply(lambda x: -np.log(max(0.01, 1 - x))) + 1

# wei_df.loc[df["death_2_Y"].eq(1), "death_2_Y"] = (df.loc[df["death_2_Y"].eq(1), "days_to_event"]/(2 * 365.25)).apply(lambda x: - np.log(x)) + 1
wei_df.loc[df["death_2_Y"].notna(), "death_2_Y"] = wei_df.loc[df["death_2_Y"].notna(), "death_2_Y"].fillna(1)
wei_df.loc[df["death_2_Y"].eq(0) & df["zero_days"].eq(0) & filter_no_immediate_hosp, "death_2_Y"] = (df.loc[df["death_2_Y"].eq(0) & df["zero_days"].eq(0), "days_to_event"]/5_000).apply(lambda x: -np.log(max(0.01, 1 - x))) + 1

# wei_df.loc[df["death_5_Y"].eq(1), "death_5_Y"] = (df.loc[df["death_5_Y"].eq(1), "days_to_event"]/(5 * 365.25)).apply(lambda x: - np.log(x)) + 1
wei_df.loc[df["death_5_Y"].eq(0) & df["zero_days"].eq(0) & filter_no_immediate_hosp, "death_5_Y"] = (df.loc[df["death_5_Y"].eq(0) & df["zero_days"].eq(0), "days_to_event"]/5_000).apply(lambda x: -np.log(max(0.01, 1 - x))) + 1
wei_df.loc[df["death_5_Y"].notna(), "death_5_Y"] = wei_df.loc[df["death_5_Y"].notna(), "death_5_Y"].fillna(1)

wei_df = wei_df.astype(float)

hosp_wei_df = pd.DataFrame(columns=targets, index=df.index)


for target_i in targets:
    # each target(1_Y,2_Y ...) should have the same amount of non-NA values
    assert df[target_i].count() == wei_df[target_i].count()

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)

pipeline = Pipeline(
    steps=[('preprocessor', median_imputer), ('classifier', model)]
)

base_model_process = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=lgb_param_grid,
    cv=None,
    scoring='accuracy',
    n_jobs=10,
    verbose=1,
    n_iter=3,
)

cv_method = KFold(n_splits=5, shuffle=True)

split_func = cv_method.split

gather_results = []

for target_i in targets:

    filter_index = df[target_i].notna()

    X = df.loc[filter_index, attr_selection]
    y = df.loc[filter_index, target_i]
    w = wei_df.loc[filter_index, target_i]

    cv_results = run_cross_validation_wei(
        split_func, pipeline, compute_binary_classification_metrics_adjusted, X, y, w
    )

    cv_results["target"] = target_i
    gather_results.append(cv_results)

results_df = pd.concat(gather_results, ignore_index=True)

