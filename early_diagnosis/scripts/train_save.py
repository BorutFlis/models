import re
import os
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from abstract_models.imputation import median_imputer

DATA_DIR = "../data"

to_train_and_save = [
    "secondary_care"
]

attr_selections = json.load(open(os.path.join(DATA_DIR, "expert_attr_selection.json")))

if "secondary_care" in to_train_and_save:
    df = pd.read_csv(os.path.join(DATA_DIR, "raw", "train.csv"), index_col=[0, 1])
    target = "Dia_MULTI"

    column_map_df = pd.read_csv(os.path.join(DATA_DIR, "column_map_df.csv"))
    predictor_categories = ["Echo", "Med", "Pat", "ECG", "Blo", "Phy", "Sym"]

    secondary_care_attr_selection = list(
        filter(
            lambda x: not re.search(".+_TP$",x),
            column_map_df.loc[column_map_df["category"].isin(predictor_categories), "cat_attr"].to_list()
        )
    )
    for attr_remove_i in ["Phy_DOB", "Pat_Oth3T", "Pat_Oth2T", "Pat_Oth1T", "Phy_DOD"]:
        secondary_care_attr_selection.remove(attr_remove_i)

    secondary_care_attr_selection.remove("Echo_SLVF")
    secondary_care_attr_selection.append("Echo_LVEF")

    df = df.dropna(subset=target)

    X = df.loc[:, df.columns.intersection(secondary_care_attr_selection)]
    y = df[target].map({"HFPEF":2, "HFREF": 1, "N": 0})

    pipeline = Pipeline(
        steps=[('imputer', median_imputer), ('model', RandomForestClassifier())]
    )
    w = pd.Series(np.where(X["Blo_NT"].notna(), 2, 1), index=X.index)

    new_pipeline = clone(pipeline)
    new_pipeline.fit(X, y, model__sample_weight=w)
    with open(os.path.join(DATA_DIR, "models", "secondary_care_early_diagnosis.pkl"), "wb") as f:
        pickle.dump(new_pipeline, f)

if "early_diagnosis" in to_train_and_save:
    df = pd.read_csv(os.path.join(DATA_DIR, "processed", "balanced_ED_NT.csv"))

    attrs = list(set(attr_selections["expert"]).intersection(df.columns)) + ['Med_LD']

    df = df.rename(columns={"Med_LD_permanent": "Med_LD"})
    df["Med_LD"] = df["Med_LD"].map({1: "Y", 0: "N"})
    X = df.loc[:, attrs]
    y = df["Dia_HFD_12M"].astype(int)
    pipeline = Pipeline(
        steps=[('imputer', median_imputer), ('model', RandomForestClassifier())]
    )
    pipeline.fit(X, y)
    with open(os.path.join(DATA_DIR, "models", "early_diagnosis.pkl"), "wb") as f:
        pickle.dump(pipeline, f)

if "early_diagnosis_HF_type" in to_train_and_save:
    df = pd.read_csv(os.path.join(DATA_DIR, "processed", "balanced_ED_NT.csv"))
    attrs = list(set(attr_selections["expert"]).intersection(df.columns)) + ['Med_LD']

    df = df.rename(columns={"Med_LD_permanent": "Med_LD"})
    df["Med_LD"] = df["Med_LD"].map({1: "Y", 0: "N"})
    df = df.dropna(subset="HF_type")
    X = df.loc[:, attrs]
    y = df["HF_type"]
    pipeline = Pipeline(
        steps=[('imputer', median_imputer), ('model', RandomForestClassifier())]
    )
    pipeline.fit(X, y)

    with open(os.path.join(DATA_DIR, "models", "early_diagnosis_HF_type.pkl"), "wb") as f:
        pickle.dump(pipeline, f)