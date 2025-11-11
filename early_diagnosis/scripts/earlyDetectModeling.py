import glob
import os
import re
import itertools
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, brier_score_loss, log_loss
from scipy.stats import entropy

def generate_decay_map(features, default_decay=0.001):
    group_decay = {     # vpliv meritve se prepolovi po:
        "_Blo": 0.02,   # 34.7 dni
        "_Phy": 0.01,   # 69.3 dni
        "_Sym": 0.01,   # 69.3 dni
        "_Pat": 0.0005, # 1386 dni
        "_Med": 0.005,  # 138.6 dni
        "_ECG": 0.02,    # 34.7 dni
        # default         693 dni
    }

    decay_map = {}

    for feature in features:
        for suffix, decay in group_decay.items():
            if feature.endswith(suffix):
                decay_map[feature] = decay
                break
        else:
            decay_map[feature] = default_decay  # če ni zadetka

    decay_map["_default_"] = default_decay
    return decay_map

def wide_to_long(df):
    index_numbers = [1, 2, 3, 4]
    sep = '_'
    id_col = 'ID'

    cols_with_index = []
    cols_without_index = []
    for col in df.columns:
        if any(f'{sep}{i}{sep}' in col for i in index_numbers):
            cols_with_index.append(col)
        else:
            cols_without_index.append(col)

    long_dfs = []
    for i in index_numbers:
        pattern = re.compile(f'{sep}{i}{sep}')
        cols_for_i = [col for col in cols_with_index if pattern.search(col)]
        renamed_cols = {col: col.replace(f'{sep}{i}{sep}', sep) for col in cols_for_i}
        df_i = df[cols_for_i].rename(columns=renamed_cols)
        df_i = pd.concat([df[cols_without_index], df_i], axis=1)
        df_i['index'] = i
        long_dfs.append(df_i)

    long_df = pd.concat(long_dfs, ignore_index=True)

    # Sortiramo po ID in index
    long_df = long_df.sort_values(by=[id_col, 'index']).reset_index(drop=True)

    # Odstranimo vrstice, kjer je TP_Phy manjkajoč
    long_df = long_df.dropna(subset=["TP_Phy"])

    # Odstranimo stolpec 'index'
    #long_df = long_df.drop(columns=["index"])

    return long_df

def add_missing_indicators(df, ignore_cols=None):
    """
    Dodaj indikatorje manjkajočih vrednosti (_is_missing) za vse stolpce,
    razen tistih v ignore_cols — tudi če stolpec nima nobenih NaN vrednosti.
    """
    if ignore_cols is None:
        ignore_cols = set()
    else:
        ignore_cols = set(ignore_cols)

    cols_to_mark = [col for col in df.columns if col not in ignore_cols]

    # Vedno ustvari indikatorje za vse izbrane stolpce
    indicators = {
        f"{col}_is_missing": df[col].isnull().astype(int)
        for col in cols_to_mark
    }

    # Združimo indikatorje z originalnim DataFrame
    df = pd.concat([df, pd.DataFrame(indicators, index=df.index)], axis=1)

    return df

def util_classify(row):
    if row["HFD_Dia"] == "N":
        return "NO_HF"
    elif row["HFD_Dia"] == "Y":
        # Poiščemo prvo razpoložljivo vrednost med SLVF_Echo, LVEF_Echo, LVEF_MRI
        for ef_col in ["SLVF_Echo", "LVEF_Echo", "LVEF_MRI"]:
            ef_value = row.get(ef_col)
            if pd.notnull(ef_value):
                if ef_value >= 50:
                    return "PRESERVED"
                else:
                    return "REDUCED"
        return None  # ali "UNKNOWN", če želiš eksplicitno vrednost
    else:
        return None

def add_class(df):
    df["Class"] = df.apply(util_classify, axis=1)
    return df

def extract_nominal_levels(attribute_names, metadata_dir):
    """
    Vrne slovar oblike {atribut_brez_indeksa: seznam_vseh_vrednosti} za nominalne atribute v long formatu,
    tudi če so v metapodatkih z indeksi (npr. Mur_1_Phy, Mur_2_Phy, ...).
    """
    suffix_to_file = {}
    loaded_metadata = {}
    attr_to_levels = defaultdict(set)

    # Zgradi mapo: "_Phy" -> "column_metadata_Phy.json"
    for fname in os.listdir(metadata_dir):
        if fname.startswith("column_metadata_") and fname.endswith(".json"):
            suffix = "_" + fname.split("_")[-1].split(".")[0]
            suffix_to_file[suffix] = os.path.join(metadata_dir, fname)

    # Preveri vsak atribut (brez indeksa)
    for attr in attribute_names:
        for suffix, filepath in suffix_to_file.items():
            if attr.endswith(suffix):
                attr_base = attr[:-len(suffix)]  # npr. "Mur" iz "Mur_Phy"

                # Naloži metapodatke, če še niso
                if suffix not in loaded_metadata:
                    with open(filepath, 'r') as f:
                        loaded_metadata[suffix] = json.load(f)

                metadata = loaded_metadata[suffix]

                # Poišči vse ujemajoče se atribute: npr. "Mur_1_Phy", "Mur_2_Phy", "Mur_Phy"
                regex = re.compile(rf"^{re.escape(attr_base)}(_\d+)?{re.escape(suffix)}$")
                for column_info in metadata:
                    col_name = column_info["column"]
                    if regex.match(col_name) and column_info.get("class") == "factor":
                        levels = column_info.get("levels", [])
                        if isinstance(levels, dict):
                            values = list(levels.values())
                        else:
                            values = levels
                        attr_to_levels[attr].update(values)
                break  # Naslednji atribut

    # Pretvori množice v sezname
    return {k: sorted(v) for k, v in attr_to_levels.items()}


def compute_time_weighted_aggregates(
    df,
    group_col,
    time_col,
    nominal_levels,
    decay=0.001,
    decay_map=None
):
    df = df.copy()
    time_series = pd.to_datetime(time_col, format="%Y-%m-%d")
    group_series = pd.Series(group_col)

    # Kombiniraj za sortiranje
    df["__group__"] = group_series.values
    df["__time__"] = time_series.values
    df.sort_values(by=["__group__", "__time__"], inplace=True)

    # Zaznaj atribute
    all_features = [col for col in df.columns if col not in ["__group__", "__time__"]]
    cat_features = [col for col in all_features if col in nominal_levels]
    num_features = [col for col in all_features if col not in nominal_levels]

    # 🛠 Dodaj stolpce v enem koraku, da se izognemo fragmentaciji
    num_new_cols = {f"{col}_twavg": np.nan for col in num_features}
    cat_new_cols = {
        f"{col}_{val}_tw": np.nan
        for col in cat_features
        for val in nominal_levels[col]
    }
    new_cols_df = pd.DataFrame({**num_new_cols, **cat_new_cols}, index=df.index)
    df = pd.concat([df, new_cols_df], axis=1)

    # Iteracija po skupinah
    for group_id, group_df in df.groupby("__group__"):
        previous_rows = []

        for idx, row in group_df.iterrows():
            current_time = row["__time__"]
            num_aggregates = {}
            cat_aggregates = {col: {} for col in cat_features}

            for past_row in previous_rows:
                dt = (current_time - past_row["__time__"]).days
                if dt <= 0:
                    continue

                # Numerični
                for col in num_features:
                    val = past_row[col]
                    d = decay_map.get(col, decay) if decay_map else decay
                    weight = np.exp(-d * dt)
                    if not pd.isnull(val):
                        num_aggregates.setdefault(col, []).append((val, weight))

                # Nominalni
                for col in cat_features:
                    val = past_row[col]
                    d = decay_map.get(col, decay) if decay_map else decay
                    weight = np.exp(-d * dt)
                    if not pd.isnull(val):
                        cat_aggregates[col][val] = cat_aggregates[col].get(val, 0) + weight

            # Shrani numerične
            for col, values in num_aggregates.items():
                if values:
                    vals, weights = zip(*values)
                    tw_avg = np.average(vals, weights=weights)
                    df.at[idx, f"{col}_twavg"] = tw_avg

            # Shrani nominalne
            for col in cat_features:
                value_weights = cat_aggregates[col]
                total_weight = sum(value_weights.values())
                for val in nominal_levels[col]:
                    col_name = f"{col}_{val}_tw"
                    if total_weight > 0 and val in value_weights:
                        df.at[idx, col_name] = value_weights[val] / total_weight
                    else:
                        df.at[idx, col_name] = 0.0

            previous_rows.append(row)

    # Odstrani pomožna stolpca
    df.drop(columns=["__group__", "__time__"], inplace=True)

    # 🧹 Defragmentiraj DataFrame
    df = df.copy()

    return df

def encode_nominals_with_known_levels(X, nominal_levels):
    """
    Vsak nominalni atribut zakodira z utežmi glede na frekvenco vrednosti.
    Če vrednost ni prisotna, doda stolpec z vrednostjo 0.
    """
    encoded_parts = []
    for col in X.columns:
        if col in nominal_levels:
            levels = nominal_levels[col]
            freqs = X[col].value_counts(normalize=True).to_dict()

            for level in levels:
                new_col = f"{col}__{level}"
                encoded_parts.append(
                    pd.Series((X[col] == level).astype(int), name=new_col)
                )
        else:
            encoded_parts.append(X[[col]])  # numerični atributi ali nominalni brez zaloge

    return pd.concat(encoded_parts, axis=1)

# ---------------------------
# 1) Train model
# ---------------------------
def train_model(train_df, selected_features, nominal_levels, decay_map):
    train_df = wide_to_long(train_df)
    train_df = add_class(train_df)
    train_df = train_df.dropna(subset=["Class"])
    X_train = train_df[selected_features]

    X_train = compute_time_weighted_aggregates(
        df=X_train,
        group_col=train_df["ID"],
        time_col=train_df["TP_Phy"],
        nominal_levels=nominal_levels,
        decay_map=decay_map
    )

    X_train = add_missing_indicators(X_train)
    y_train = train_df["Class"]

    X_train = encode_nominals_with_known_levels(X_train, nominal_levels)

    if y_train.dtype == 'object':
        encoder = LabelEncoder()
        y_train_enc = encoder.fit_transform(y_train)
        class_labels = encoder.classes_
    else:
        encoder = None
        y_train_enc = y_train
        class_labels = np.unique(y_train_enc)

    model = xgb.XGBClassifier(
        eval_metric='mlogloss',
        objective='multi:softprob',
        num_class=len(class_labels),
        missing=np.nan
    )
    model.fit(X_train, y_train_enc)

    return model, encoder, class_labels


# ---------------------------
# 2) Predict with model
# ---------------------------
def predict_with_model(model, data_df, selected_features, nominal_levels, decay_map, encoder):
    df = wide_to_long(data_df)
    df = add_class(df)
    df = df.dropna(subset=["Class"])
    X = df[selected_features]

    X = compute_time_weighted_aggregates(
        df=X,
        group_col=df["ID"],
        time_col=df["TP_Phy"],
        nominal_levels=nominal_levels,
        decay_map=decay_map
    )

    X = add_missing_indicators(X)
    y_true = df["Class"]

    X = encode_nominals_with_known_levels(X, nominal_levels)

    if encoder is not None:
        y_true_enc = encoder.transform(y_true)
    else:
        y_true_enc = y_true

    y_pred_enc = model.predict(X)
    y_proba = model.predict_proba(X)

    if encoder is not None:
        y_true_labels = encoder.inverse_transform(y_true_enc)
        y_pred_labels = encoder.inverse_transform(y_pred_enc)
        class_labels = encoder.classes_
    else:
        y_true_labels = y_true_enc
        y_pred_labels = y_pred_enc
        class_labels = np.unique(y_true_enc)

    preds_df = pd.DataFrame({
        "ID": df["ID"].values,
        "TrueClass": y_true_labels,
        "PredClass": y_pred_labels,
    })

    for idx, cl in enumerate(class_labels):
        preds_df[f"Prob_Class_{cl}"] = y_proba[:, idx]

    return preds_df


# ---------------------------
# 3) Threshold tuning (per-class)
# ---------------------------
def threshold_tuning(preds_m1, preds_m2, threshold_grid):
    """
    threshold_grid: dict {class_label: list_of_thresholds}
    """
    classes = list(threshold_grid.keys())
    grids = [threshold_grid[c] for c in classes]

    best_acc = -1
    best_thresholds = None

    for combination in itertools.product(*grids):
        print(combination)
        thr_map = dict(zip(classes, combination))
        combined_pred = []

        for idx, row in preds_m1.iterrows():
            true_class = row["TrueClass"]
            prob_m1 = row[f"Prob_Class_{true_class}"]
            thr = thr_map[true_class]
            if prob_m1 >= thr:
                pred_class = preds_m1.loc[idx, "PredClass"]
            else:
                pred_class = preds_m2.loc[idx, "PredClass"]
            combined_pred.append(pred_class)

        acc = accuracy_score(preds_m1["TrueClass"], combined_pred)
        if acc > best_acc:
            best_acc = acc
            best_thresholds = thr_map

    return best_thresholds, best_acc


# ---------------------------
# 4) Evaluate with thresholds
# ---------------------------
def evaluate(preds_m1, preds_m2, thresholds_per_class):
    prob_cols = [c for c in preds_m1.columns if c.startswith("Prob_Class_")]
    class_labels = np.array([c.replace("Prob_Class_", "") for c in prob_cols])
    n_classes = len(class_labels)

    # preveri skladnost stolpcev
    missing_in_m2 = [c for c in prob_cols if c not in preds_m2.columns]
    if missing_in_m2:
        raise ValueError(f"Manjkajoči stolpci v preds_m2: {missing_in_m2}")

    combined_pred = []
    probs_final = []
    count_m1_only = 0
    count_m2_used = 0

    # zanka po vseh primerih
    for idx, row in preds_m1.iterrows():
        true_class = row["TrueClass"]
        prob_m1_true = row[f"Prob_Class_{true_class}"]
        thr = thresholds_per_class[true_class]

        if prob_m1_true >= thr:
            # uporabi model 1
            pred_class = row["PredClass"]
            prob_vector = row[prob_cols].astype(float).values
            count_m1_only += 1
        else:
            # uporabi model 2
            pred_class = preds_m2.loc[idx, "PredClass"]
            prob_vector = preds_m2.loc[idx, prob_cols].astype(float).values
            count_m2_used += 1

        combined_pred.append(pred_class)
        probs_final.append(prob_vector)

    probs_final = np.vstack(probs_final)

    # y_true za metrike
    y_true = preds_m1["TrueClass"].values

    # metrike
    acc = accuracy_score(y_true, combined_pred)
    f1 = f1_score(y_true, combined_pred, average='macro')
    cm = confusion_matrix(y_true, combined_pred, labels=class_labels)
    logloss = log_loss(y_true, probs_final, labels=class_labels)

    # Brier score
    idx_map = {lab: i for i, lab in enumerate(class_labels)}
    y_idx = np.array([idx_map[y] for y in y_true])
    y_onehot = np.zeros_like(probs_final)
    y_onehot[np.arange(len(y_idx)), y_idx] = 1.0
    brier = np.mean(np.sum((probs_final - y_onehot) ** 2, axis=1))

    return {
        "accuracy": acc,
        "f1": f1,
        "brier": brier,
        "log_loss": logloss,
        "confusion_matrix": cm,
        "count_m1_only": count_m1_only,
        "count_m2_used": count_m2_used,
        "combined_predictions": combined_pred
    }

# ---------------------------
# 5) Main pipeline
# ---------------------------
def run_cross_validation_pipeline(data_dir, reduced_features, full_features, nominal_levels, decay_map, threshold_grid):
    files = glob.glob(os.path.join(data_dir, "fold_*_train.csv"))
    fold_indices = sorted(set(int(f.split("_")[-2]) for f in files))

    results = []

    all_true = []
    all_pred = []

    for i in fold_indices:
        print(f"\n🌀 Fold {i}")

        train_file = os.path.join(data_dir, f"fold_{i}_train.csv")
        test_file = os.path.join(data_dir, f"fold_{i}_test.csv")

        train_df = pd.read_csv(train_file, low_memory=False)
        test_df = pd.read_csv(test_file, low_memory=False)

        # Train both models
        print("ucenje prvega modela")
        model1, encoder1, classes = train_model(train_df, reduced_features, nominal_levels, decay_map)

        print("ucenje drugega modela")
        model2, encoder2, _ = train_model(train_df, full_features, nominal_levels, decay_map)

        # Predict on train set for threshold tuning
        print("napovedi prvega modela")
        preds_train_m1 = predict_with_model(model1, train_df, reduced_features, nominal_levels, decay_map, encoder1)

        print("napovedi drugega modela")
        preds_train_m2 = predict_with_model(model2, train_df, full_features, nominal_levels, decay_map, encoder2)

        # Tune thresholds
        print("izbira praga za preklop med modeli")
        best_thresholds, best_train_acc = threshold_tuning(preds_train_m1, preds_train_m2, threshold_grid)

        print(f"📌 Best thresholds: {best_thresholds} | Train acc: {best_train_acc:.3f}")

        # Predict on test set
        preds_test_m1 = predict_with_model(model1, test_df, reduced_features, nominal_levels, decay_map, encoder1)
        preds_test_m2 = predict_with_model(model2, test_df, full_features, nominal_levels, decay_map, encoder2)

        # Evaluate
        metrics = evaluate(preds_test_m1, preds_test_m2, best_thresholds)

        print(f"✅ Test accuracy: {metrics['accuracy']:.3f} | "
              f"F1: {metrics['f1']:.3f} | "
              f"Brier: {metrics['brier']:.4f} | "
              f"LogLoss: {metrics['log_loss']:.4f}")
        print(f"📊 Model1-only: {metrics['count_m1_only']} | "
              f"Model2-used: {metrics['count_m2_used']} "
              f"({metrics['count_m2_used'] / (metrics['count_m1_only'] + metrics['count_m2_used']):.1%} primerov)")
        print(f"🔍 Confusion Matrix:\n{metrics['confusion_matrix']}")

        # Shrani rezultate folda
        results.append({
            "fold": i,
            "train_acc": best_train_acc,
            "test_acc": metrics['accuracy'],
            "test_f1": metrics['f1'],
            "test_brier": metrics['brier'],
            "test_log_loss": metrics['log_loss'],
            "count_m1_only": metrics['count_m1_only'],
            "count_m2_used": metrics['count_m2_used'],
            "thresholds": best_thresholds
        })

        # Shrani za globalno matriko zmot
        all_true.extend(preds_test_m1["TrueClass"].tolist())
        all_pred.extend(metrics["combined_predictions"])

    # Končna matrika zmot
    final_cm = confusion_matrix(all_true, all_pred)

    print("\n📊 Končna matrika zmot čez vse folde:")
    print(final_cm)

    df_results = pd.DataFrame(results)
    return df_results, final_cm


def do_all():
    reduced_features = [
        "Sex_Phy", "Hei_Phy", "Age_Phy", "Wei_Phy", "BMI_Phy", "Mur_Phy", "BPO_Phy",
        "PMI_Pat", "CABG_Pat", "PTCA_Pat", "DM_Pat", "Hyp_Pat", "AF_Pat",
        "LD_Med",
        "DAR_Sym", "Ort_Sym", "TAR_Sym", "PO_Sym", "DWSU_Sym", "CPR_Sym", "Pal_Sym", "PC_Sym", "PlE_Sym", "PrE_Sym",
        "VC_Sym",
        "NT_Blo", "BNPa_Blo", "BNPc_Blo", "EGFR_Blo"
    ]

    full_features = [
        "Sex_Phy", "Hei_Phy", "Age_Phy", "Wei_Phy", "BMI_Phy", "Mur_Phy", "BPO_Phy",
        "PMI_Pat", "CABG_Pat", "PTCA_Pat", "DM_Pat", "Hyp_Pat", "AF_Pat",
        "LD_Med",
        "DAR_Sym", "Ort_Sym", "TAR_Sym", "PO_Sym", "DWSU_Sym", "CPR_Sym", "Pal_Sym", "PC_Sym", "PlE_Sym", "PrE_Sym",
        "VC_Sym",
        "NT_Blo", "BNPa_Blo", "BNPc_Blo", "EGFR_Blo",
        "AFib_ECG", "AFlu_ECG", "QRSD_ECG", "CLBBB_ECG", "STSE_ECG", "STSD_ECG", "SQW_ECG", "STSA_ECG", "NTW_ECG",
        "QTD_ECG", "LVH_ECG", "HF_ECG", "HA_ECG",
        "HF_Echo", "AF_Echo", "LVIDd_Echo", "LVIDs_Echo", "LVEDV_Echo", "LVESV_Echo", "EVM_Echo", "AVM_Echo",
        "EDD_Echo", "IVS_Echo", "PWD_Echo", "LAD_Echo", "LVH_Echo", "LAVs_Echo", "MR_Echo", "SVLV_Echo", "Hypo_Echo",
        "Aki_Echo", "Dys_Echo", "EE_Echo", "DDG_Echo", "TAPSE_Echo", "RVSP_Echo", "AR_Echo", "TR_Echo", "LVEF_Echo"
    ]

    all_features = list(dict.fromkeys(reduced_features + full_features))
    decay_map = generate_decay_map(all_features)

    cv_data_dir = "data\\CV_folds"  # <-- spremeni to pot!
    metadata_dir = "data"

    nominal_levels = extract_nominal_levels(all_features, metadata_dir)

    # Define threshold search grid per class
    # threshold_grid = {
    #     'NO_HF': np.linspace(0.3, 0.7, 5),
    #     'REDUCED': np.linspace(0.3, 0.7, 5),
    #     'PRESERVED': np.linspace(0.3, 0.7, 5)
    # }

    threshold_grid = {
        'NO_HF': np.linspace(1.0, 1.0, 1),
        'REDUCED': np.linspace(1.0, 1.0, 1),
        'PRESERVED': np.linspace(1.0, 1.0, 1)
    }

    results, final_cm = run_cross_validation_pipeline(cv_data_dir, reduced_features, full_features, nominal_levels, decay_map, threshold_grid)

    # Povprečne metrike čez vse fold-e
    print(results.mean(numeric_only=True))

    # Posamezni pragi po fold-ih
    for fold, thr in zip(results['fold'], results['thresholds']):
        print(f"Fold {fold} thresholds:", thr)

    # Shrani v CSV za kasnejšo analizo
    results.to_csv("m1_cv_results.csv", index=False)

    print("Končna matrika zmot čez vse fold-e:")
    print(final_cm)


do_all()

