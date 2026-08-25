import os

import pandas as pd

DATA_DIR = "../data"
DATA_DUMP_DIR = "../data_dump"

df = pd.read_csv(os.path.join(DATA_DIR, "raw", "risk_stratification_sample.csv"), index_col=[0, 1])
df = df.reset_index()
df["Timepoint"] = pd.to_datetime(df["Timepoint"])
df["days_to_HFD_new"] = df.groupby("ID", sort=False)["Timepoint"].apply(lambda x: (x.iat[-1] - x).dt.days).droplevel(level=0)

df["2Y"] = pd.Series()
df["5Y"] = pd.Series()
df["10Y"] = pd.Series()

df.loc[df["Dia_HFD_patient"].eq(0), "days_to_censoring"] = df.loc[df["Dia_HFD_patient"].eq(0), "days_to_HFD_new"]
for year_i in (2, 5, 10):
    df.loc[df["Dia_HFD_patient"].eq(1) & df["days_to_HFD"].le(year_i * 365.25), f"{year_i}Y"] = "Y"
    df.loc[df["Dia_HFD_patient"].eq(1) & df["days_to_HFD"].gt(year_i * 365.25), f"{year_i}Y"] = "N"

    df.loc[df["Dia_HFD_patient"].eq(0) & df["days_to_censoring"].gt(year_i * 365.25), f"{year_i}Y"] = "N"

# test all that are Y in 2 years should Y in 5, 10 years and all Y in 5 years should be Y in 10 years
assert df.loc[df["2Y"].eq("Y"), ["5Y", "10Y"]].eq("Y").all().all()
# test Dia_HFD_patient(=1) should not have NA values in 2Y, 5Y, 10Y
assert df.loc[df["Dia_HFD_patient"].eq(1), ["2Y", "5Y", "10Y"]].notna().all().all()
# test Dia_HFD_patient(=0) should have more N values in 2Y than 5Y, 10Y
assert df.loc[df["Dia_HFD_patient"].eq(0), ["2Y","5Y", "10Y"]].count().is_monotonic_decreasing

already_hfd_filter = df["days_to_HFD"].lt(30)
df = df.drop(df.index[already_hfd_filter])

# med_cols = ["Med_Sta_permanent", "Med_Sta_issued_this_month", "Med_LD_permanent", "Med_LD_issued_this_month"]
# df.loc[:, med_cols] = df.loc[:, med_cols].map(lambda x: {1.0:"Y", 0.0:"N"}.get(x, "N")).astype(str)

to_drop = [
    'days_to_censoring', 'days_to_HFD_new', 'days_in_observation', 'Dia_HFD_patient', 'Dia_HFD_event',
    'days_to_HFD'
]
df = df.drop(to_drop, axis=1)

df = df.set_index(["ID", "Timepoint"])
df.to_csv(os.path.join(DATA_DIR, "processed", "risk_stratification.csv"))
