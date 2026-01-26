import os

import pandas as pd

from abstract_models.metric_utils import binary_metrics_by_age_cohort

DATA_DIR = "../data"

age_balanced_df = pd.read_csv(os.path.join(DATA_DIR, "processed", "age_balanced_ED_NT.csv"), index_col=[0, 1])
balanced_df = pd.read_csv(os.path.join(DATA_DIR, "processed", "balanced_ED_NT.csv"), index_col=[0, 1])
df = pd.read_csv(os.path.join(DATA_DIR, "processed", "early_diagnosis_NT.csv"), index_col=[0, 1])

rest_of_results_df = pd.read_csv(os.path.join(DATA_DIR, "results", "ED_rest_of.csv"), index_col=[0, 1])