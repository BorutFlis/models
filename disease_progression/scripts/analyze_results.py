import os
import re

import pandas as pd
import numpy as np

from abstract_models.survival import chunk_cindex_calculations

DATA_DIR = "../data"
RESULTS_DIR = os.path.join(DATA_DIR, "results")

DATA_DUMP_DIR = "../data_dump"
df = pd.read_csv(os.path.join(DATA_DIR, "processed", "high_risk_HES_big.csv"))
results_df = pd.read_csv(os.path.join(RESULTS_DIR, "wei_type_classification.csv"))

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

gather_cindex = []
for wei_type, wei_pred_df in weighted_preds_container.items():
    for target in [ 'death_1_Y', 'death_2_Y', 'death_5_Y']:

        try:
            concordant, discordant, t1, t2 = chunk_cindex_calculations(
                wei_pred_df.loc[:, ["patid", "date", target, "days_to_event", "death_event"]].dropna(subset=target).rename(columns={target: "estimate"}),
                chunk=2000,
                filter_func=None
            )
            cindex_i = concordant/(concordant + discordant)
        except AssertionError:
            print(f"{wei_type}, {target} problem with counting")
            continue
        gather_cindex.append(
            {"wei_type": wei_type, "target": target, "cindex": cindex_i}
        )

cindex_df = pd.DataFrame(gather_cindex)
