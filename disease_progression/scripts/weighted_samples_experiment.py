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

small_test = True

DATA_DIR = "../data"
DATA_DUMP_DIR = "../data_dump"
RESULTS_DIR = os.path.join(DATA_DIR, "results")

# "zero_days",
target_container = ['death_1_Y', 'death_2_Y', 'death_5_Y']

schema_df = pd.read_csv(os.path.join(DATA_DIR, "processed", "high_risk_HES_big_schema.csv"))

attr_selection = schema_df.loc[schema_df["variable_type"].eq("predictor"), "attr"].to_list()

df = pd.read_csv(os.path.join(DATA_DIR, "processed", "high_risk_HES_big.csv"))
df["date"] = pd.to_datetime(df["date"])
df["regstartdate"] = pd.to_datetime(df["regstartdate"])
df = df.loc[(df["date"] - df["regstartdate"]).dt.days.gt(365)]
df = df.set_index(["patid", "date"])

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
df = df.iloc[np.random.choice(len(df), 1000, replace=False)]

gather_majority_class = []
for target_i in target_container:
    gather_majority_class.append(
        {"target": target_i, "positive_proportion": df[target_i].value_counts(normalize=True)[1.0]}
    )

class_stats_df = pd.DataFrame(gather_majority_class)

#if small_test:
#    df = df.iloc[np.random.choice(len(df), 1000, replace=False)]

filter_no_immediate_hosp = df['post_emmergency_hosp_n_in_90_days'].isna() | df['post_emmergency_hosp_n_in_90_days'].eq(0)

wei_df = pd.DataFrame(columns=target_container, index=df.index)
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

normal_wei_df = pd.DataFrame(columns=target_container, index=df.index)
normal_wei_df.loc[df["death_1_Y"].notna(), "death_1_Y"] = normal_wei_df.loc[df["death_1_Y"].notna(), "death_1_Y"].fillna(1)
normal_wei_df.loc[df["death_2_Y"].notna(), "death_2_Y"] = normal_wei_df.loc[df["death_2_Y"].notna(), "death_2_Y"].fillna(1)
normal_wei_df.loc[df["death_5_Y"].notna(), "death_5_Y"] = normal_wei_df.loc[df["death_5_Y"].notna(), "death_5_Y"].fillna(1)

multiplier = 3

high_risk_wei_df = normal_wei_df.copy()
high_risk_filter = df["zero_days"].eq(1) & df["days_to_event"].lt(90)
high_risk_wei_df.loc[high_risk_filter, "death_1_Y"] = high_risk_wei_df.loc[high_risk_filter, "death_1_Y"] * multiplier
high_risk_wei_df.loc[high_risk_filter, "death_2_Y"] = high_risk_wei_df.loc[high_risk_filter, "death_2_Y"] * multiplier
high_risk_wei_df.loc[high_risk_filter, "death_5_Y"] = high_risk_wei_df.loc[high_risk_filter, "death_5_Y"] * multiplier

low_risk_wei_df = normal_wei_df.copy()
low_risk_filter = filter_no_immediate_hosp & df["zero_days"].eq(0) & df["days_to_event"].gt(1000)
low_risk_wei_df.loc[low_risk_filter, "death_1_Y"] = low_risk_wei_df.loc[low_risk_filter, "death_1_Y"] * multiplier
low_risk_wei_df.loc[low_risk_filter, "death_2_Y"] = low_risk_wei_df.loc[low_risk_filter, "death_2_Y"] * multiplier
low_risk_wei_df.loc[low_risk_filter, "death_5_Y"] = low_risk_wei_df.loc[low_risk_filter, "death_5_Y"] * multiplier

clear_wei_df = normal_wei_df.copy()

clear_wei_df.loc[high_risk_filter, "death_1_Y"] = clear_wei_df.loc[high_risk_filter, "death_1_Y"] * multiplier
clear_wei_df.loc[high_risk_filter, "death_2_Y"] = clear_wei_df.loc[high_risk_filter, "death_2_Y"] * multiplier
clear_wei_df.loc[high_risk_filter, "death_5_Y"] = clear_wei_df.loc[high_risk_filter, "death_5_Y"] * multiplier

clear_wei_df.loc[low_risk_filter, "death_1_Y"] = clear_wei_df.loc[low_risk_filter, "death_1_Y"] * multiplier
clear_wei_df.loc[low_risk_filter, "death_2_Y"] = clear_wei_df.loc[low_risk_filter, "death_2_Y"] * multiplier
clear_wei_df.loc[low_risk_filter, "death_5_Y"] = clear_wei_df.loc[low_risk_filter, "death_5_Y"] * multiplier

wei_container = {
    "normal": normal_wei_df,
    "low_risk_weighted": low_risk_wei_df,
    "high_risk_weighted": high_risk_wei_df,
    "clear_examples_weighted": clear_wei_df
}
#, "clear_example_weighted": pd.DataFrame()}

gather_wei_class_stats = []
for wei_type, wei_df in wei_container.items():
    test_df = pd.merge(wei_df, df.loc[:, target_container], suffixes=("", "_label"), left_index=True, right_index=True)
    for target_i in target_container:
        wei_value_counts_df = test_df.groupby(f"{target_i}_label")[target_i].sum()
        gather_wei_class_stats.append({
            "wei": wei_type,
            "target": target_i,
            "weighted_positive_proportion": wei_value_counts_df[1.0] / wei_value_counts_df.sum()
        })
wei_class_stats_df = pd.DataFrame(gather_wei_class_stats)


for target_i in target_container:
    # each target(1_Y,2_Y ...) should have the same amount of non-NA values
    assert df[target_i].count() == wei_df[target_i].count()

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbose=-1
)

pipeline = Pipeline(
    steps=[('preprocessor', median_imputer), ('classifier', model)]
)

base_model_process = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=lgb_param_grid,
    cv=None,
    scoring='accuracy',
    n_jobs=100,
    verbose=1,
    n_iter=100,
)

cv_method = KFold(n_splits=5, shuffle=True)

split_func = cv_method.split

gather_results = []
gather_predictions = {}

for wei_type, wei_df in wei_container.items():
    gather_predictions[wei_type] = []

    for target_i in target_container:
        print(f"{wei_type}, {target_i}")
        filter_index = df[target_i].notna()

        X = df.loc[filter_index, attr_selection]
        y = df.loc[filter_index, target_i]
        w = wei_df.loc[filter_index, target_i]

        cv_results, predictions = run_cross_validation_wei_preds(
            split_func, pipeline, compute_binary_classification_metrics_adjusted, X, y, w
        )
        predictions.rename(columns={0: target_i}, inplace=True)

        cv_results["target"] = target_i
        cv_results["wei"] = wei_type
        gather_results.append(cv_results)
        gather_predictions[wei_type].append(predictions)

results_df = pd.concat(gather_results, ignore_index=True)
results_df = pd.merge(results_df, class_stats_df, on="target")
results_df = pd.merge(results_df, wei_class_stats_df, left_on=("wei", "target"), right_on=("wei", "target"))
results_df.to_csv(os.path.join(RESULTS_DIR, "wei_type_classification.csv"), index=False)

for k, df_container in gather_predictions.items():
    wei_pred_df = pd.concat(df_container, axis=1)
    wei_pred_df.to_csv(os.path.join(DATA_DIR, "interim", f"{k}_preds.csv"))

#
# estimates_df = pd.concat(gather_predictions, axis=1)
# estimates_event_df = pd.merge(estimates_df, df.loc[:, ["days_to_event", "zero_days"]], left_index=True, right_index=True).rename(columns={"zero_days": "death_event"})
#
# sample_estimates_event_df = estimates_event_df.iloc[np.random.choice(len(estimates_event_df), 1000, replace=False)]
#
#
