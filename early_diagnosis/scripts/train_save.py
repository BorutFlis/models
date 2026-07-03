import os
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from abstract_models.imputation import median_imputer

DATA_DIR = "../data"

to_train_and_save = [
    "early_diagnosis_HF_type"
]

attr_selections = json.load(open(os.path.join(DATA_DIR, "expert_attr_selection.json")))

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