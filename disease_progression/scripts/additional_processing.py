import os

import numpy as np
import pandas as pd


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


DATA_DIR = "../data"

mortality_data = pd.read_csv(os.path.join(DATA_DIR, "raw", "surv_ind.csv"), index_col=0)

mortality_data = year_stratification_labels(mortality_data)
mortality_data.to_csv(os.path.join(DATA_DIR, "processed", "classification.csv"))
