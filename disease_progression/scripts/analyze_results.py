import os
import re

import pandas as pd

DATA_DIR = "../data"
DATA_DUMP_DIR = "../data"

result_files = [
    'RandomForest_classifcation_dp_death_10_Y',
    'RandomForest_classifcation_dp_death_2_Y',
    'RandomForest_classifcation_dp_death_5_Y'
]

gather_results = []
for f in result_files:
    year = int(f.split("_")[-2])
    gather_results.append(
        pd.read_csv(os.path.join(DATA_DIR, "results", f)).assign(year=year)
    )


total_results_df = pd.concat(gather_results)
results_by_year_df = pd.concat([
    total_results_df.groupby("year")[
        ["Accuracy", "AUC", 'Precision', 'Sensitivity (Recall)', 'Specificity', 'F1 Score']].mean(),
    total_results_df.groupby("year")[["n_positive", "n_total"]].sum()
], axis=1)
