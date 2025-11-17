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

gather_all_dfs = []
all_files_path = os.path.join(DATA_DIR, "raw", "all_files")
for f in os.listdir(os.path.join(DATA_DIR, "raw", "all_files")):
    gather_all_dfs.append(
        pd.read_csv(os.path.join(all_files_path, f), index_col=0)
    )

df = pd.concat(gather_all_dfs)

hfref_df = pd.read_csv(os.path.join(DATA_DIR, "raw", "hfref_confirmed_HF.csv"))
hfref_df = hfref_df.set_index("patid")
df = pd.merge(df, hfref_df.loc[:, ["obsdate"]].add_prefix("hfref_"), left_index=True, right_index=True, how="left")

hfpef_df = pd.read_csv(os.path.join(DATA_DIR, "raw", "hfpef_confirmed_HF.csv"))
hfpef_df = hfpef_df.set_index("patid")
df = pd.merge(df, hfpef_df.loc[:, ["obsdate"]].add_prefix("hfpef_"), left_index=True, right_index=True, how="left")
df["HF_type"] = df["hfpef_obsdate"].mask(df["hfpef_obsdate"].notna(), "HFPEF").mask(df["hfref_obsdate"].notna(), "HFREF")
df.loc[df["hfpef_obsdate"].notna() & df["hfref_obsdate"].notna(), "HF_type"] = np.nan

df = df.drop(["hfref_obsdate", "hfpef_obsdate"], axis=1)

df = year_stratification_labels(df)

df.to_csv(os.path.join(DATA_DIR, "processed", "classification.csv"))
