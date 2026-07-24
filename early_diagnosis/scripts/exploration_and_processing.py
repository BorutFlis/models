import os

import pandas as pd

from abstract_models.metric_utils import binary_metrics_by_age_cohort, mean_std_metrics_output

DATA_DIR = "../data"

try:
    age_balanced_df = pd.read_csv(os.path.join(DATA_DIR, "processed", "age_balanced_ED_NT.csv"), index_col=[0, 1])
except FileNotFoundError:
    print("age balanced processed file not found on this computer")

balanced_df = pd.read_csv(os.path.join(DATA_DIR, "processed", "balanced_ED_NT.csv"), index_col=[0, 1])

try:
    df = pd.read_csv(os.path.join(DATA_DIR, "processed", "early_diagnosis_NT.csv"), index_col=[0, 1])
except FileNotFoundError:
    print("early diagnosis file not found on this computer")

try:
    rest_of_results_df = pd.read_csv(os.path.join(DATA_DIR, "results", "ED_rest_of.csv"), index_col=[0, 1])
except FileNotFoundError:
    print("rest of results file not found on this computer")

try:
    results_df = pd.read_csv(os.path.join(DATA_DIR, "results", "complete_results_ED.csv"))
except FileNotFoundError:
    print("complete results file not found on this computer")

output_df = mean_std_metrics_output(results_df.rename(columns={"Sensitivity": "Sensitivity (Recall)"}), groupby_col="model")

