import os
import json
import re

import pandas as pd
import numpy as np

from abstract_models.imputation import median_imputer, median_imputer_missing, ffill_median_imputer, ForwardFillImputer


def read_ml_dataset(path: str, sheet_name="Risk Stratification"):
    folder_path, file_name = os.path.split(path)
    file_ext = file_name.split(".")[-1]
    pattern = f"(\d+)\.{file_ext}$"
    m = re.search(pattern, file_name)
    file_number = int(m.groups()[0])
    id_map = json.load(open(os.path.join(folder_path, "id_maps", f"{file_number}.json"), "rt"))

    df = pd.read_csv(path, index_col=[0, 1])
    df = df.reset_index()
    df["Timepoint"] = pd.to_datetime(df["Timepoint"])
    df = df.set_index(["ID", "Timepoint"])

    new_index = pd.MultiIndex.from_tuples(
        [(int(id_map[str(step_id)]), step_date) for step_id, step_date in df.index],
        names=["ID", "date"]
    )
    df.index = new_index
    return df


ffill_imputer = ForwardFillImputer(fill_medians=False)

DATA_DIR = "../data"
raw_file_path = os.path.join(DATA_DIR, "raw", "cprd_NT")
raw_file_container = os.listdir(raw_file_path)
raw_file_container.remove("id_maps")

test_files = ["5.csv", "16.csv"]

gather_dfs = []
for i, f in enumerate(raw_file_container):
    print(f"{i}/{len(raw_file_container)}")
    df = read_ml_dataset(os.path.join(raw_file_path, f))
    df = df.drop(df.index[df["days_to_HFD"].lt(0)])
    days_in_db = df.groupby("ID")["days_to_HFD"].transform(
        lambda x: (x.index.get_level_values("date")[-1] - x.index.get_level_values("date")).days
    )
    df["days_in_db"] = np.where(df["days_to_HFD"].isna(), days_in_db, float("nan"))

    nt_filter = df["Blo_NT"].notna()

    imputed_df = ffill_imputer.fit_transform(df)
    imputed_df["Phy_Hei"] = (
        imputed_df.groupby("ID")["Phy_Hei"].apply(lambda x: x.ffill(limit=None)).droplevel(level=0)
    )
    gather_dfs.append(imputed_df.loc[nt_filter])

full_df = pd.concat(gather_dfs)
try:
    full_df = full_df.drop("Echo_LVEF", axis=1)
except KeyError:
    pass
try:
    full_df = full_df.drop("Blo_EGFR", axis=1)
except KeyError:
    pass

full_df.loc[:, full_df.columns.str.startswith("Sym_")] = (
    full_df.loc[:, full_df.columns.str.startswith("Sym_")].fillna("N")
)

assert full_df.loc[full_df["days_in_db"].notna(), "Dia_HFD_patient"].eq(0).all()


for months in [6, 12, 18]:
    step_target_attr = f"Dia_HFD_{months}M"
    full_df[step_target_attr] = pd.Series()

    full_df.loc[full_df["days_to_HFD"].le(months*30.5), step_target_attr] = 1
    full_df.loc[full_df["days_to_HFD"].gt(months*30.5), step_target_attr] = 0

    # only the ones that have long enough time horizon
    full_df.loc[full_df["days_in_db"].gt(months*30.5), step_target_attr] = 0

    assert full_df.loc[full_df["days_to_HFD"].notna(), step_target_attr].squeeze().notna().all()
    assert full_df.loc[full_df[step_target_attr].isna(), "days_in_db"].squeeze().le(months*30.5).all()

med_cols = ["Med_Sta_permanent", "Med_Sta_issued_this_month", "Med_LD_permanent", "Med_LD_issued_this_month"]
full_df.loc[:, med_cols] = full_df.loc[:, med_cols].fillna(0)
