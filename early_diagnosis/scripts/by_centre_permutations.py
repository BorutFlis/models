import os
import json
import itertools

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from abstract_models.imputation import median_imputer


DATA_DIR = "../data"
DATA_DUMP_DIR = "../data_dump"

attr_selections = json.load(open(os.path.join(DATA_DIR, "expert_attr_selection.json")))


# ----------------------------------------------------------
# User settings
# ----------------------------------------------------------
DATA_PATH = os.path.join(DATA_DIR, "raw", "train.csv")     # <-- change this
TARGET = "Dia_HFD"              # Binary classification target
CENTRE_COL = "centre"           # Centre attribute column
TEST_SIZE = 0.2
RANDOM_STATE = 42
# ----------------------------------------------------------

# Load dataset
df = pd.read_csv(DATA_PATH, index_col=[0, 1])
df = df.dropna(subset=TARGET)

# Check
if CENTRE_COL not in df.columns:
    raise ValueError(f"Column '{CENTRE_COL}' not found.")

if TARGET not in df.columns:
    raise ValueError(f"Column '{TARGET}' not found.")

# Identify all centre values
centre_values = sorted(df[CENTRE_COL].unique())
print("Centres found:", centre_values)

results = []

model = RandomForestClassifier()
imputer = median_imputer

pipeline = Pipeline(steps=[('preprocessor', imputer), ('classifier', model)])

# For all centre pairs (combinations of size 2)
for c1, c2 in itertools.combinations(centre_values, 2):

    # Extract data for both centres
    df_c1 = df[df[CENTRE_COL] == c1]
    df_c2 = df[df[CENTRE_COL] == c2]

    attrs = list(set(attr_selections["expert"]).intersection(df.columns))

    X_full = df.loc[:, attrs]
    y = df[TARGET].map({"Y": 1, "N": 0})

    # Map back to centre-specific splits
    X_c1 = X_full.loc[df_c1.index]
    y_c1 = y.loc[df_c1.index]

    X_c2 = X_full.loc[df_c2.index]
    y_c2 = y.loc[df_c2.index]

    # Skip if too small
    if len(X_c1) < 5 or len(X_c2) < 5:
        continue

    if y_c1.nunique() > 1:
        # ----------------------
        # Train on C1 → Test on C2
        # ----------------------

        pipeline.fit(X_c1, y_c1)
        preds = pipeline.predict(X_c2)
        acc_c1_to_c2 = accuracy_score(y_c2, preds)

        results.append({
            "Train Centre": c1,
            "Test Centre": c2,
            "Accuracy": acc_c1_to_c2
        })

    if y_c2.nunique() > 1:
        # ----------------------
        # Train on C2 → Test on C1
        # ----------------------
        pipeline.fit(X_c2, y_c2)
        preds = pipeline.predict(X_c1)
        acc_c2_to_c1 = accuracy_score(y_c1, preds)

        results.append({
            "Train Centre": c2,
            "Test Centre": c1,
            "Accuracy": acc_c2_to_c1
        })

# Convert to DataFrame
results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)

print("\n=== CENTRE-TO-CENTRE RESULTS ===")
print(results_df.to_string(index=False))
