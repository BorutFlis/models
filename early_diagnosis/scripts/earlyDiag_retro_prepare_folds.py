import pandas as pd
import json
import os
from sklearn.model_selection import StratifiedKFold

def apply_column_metadata(df: pd.DataFrame, json_files: list) -> pd.DataFrame:
    """
    Popravi tipe stolpcev v df glede na metapodatke v več JSON datotekah.
    Obenem preveri, ali so se pojavile nove manjkajoče vrednosti.

    Args:
        df (pd.DataFrame): izvorni DataFrame.
        json_files (list): seznam poti do JSON datotek s stolpci in tipi.

    Returns:
        pd.DataFrame: DataFrame s popravljeno tipizacijo stolpcev.
    """

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        for col_meta in meta:
            col = col_meta.get('column')
            r_class = col_meta.get('class')
            levels = col_meta.get('levels')
            ordered = col_meta.get('ordered', False)

            if col not in df.columns:
                print(f"⚠️  Stolpec '{col}' ni najden v DataFrame-u.\n")
                continue

            # Shrani masko manjkajočih vrednosti pred spremembo
            na_before = df[col].isna()

            # Poskusi pretvorbo glede na R razred
            if r_class == "factor":
                if levels is not None:
                    df[col] = pd.Categorical(df[col], categories=levels, ordered=ordered)
                else:
                    df[col] = df[col].astype("category")

            elif r_class == "Date":
                dayfirst = col_meta.get("dayfirst", False)
                df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=dayfirst)
                # df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")

            elif r_class == "numeric":
                df[col] = pd.to_numeric(df[col], errors="coerce")

            elif r_class == "integer":
                df[col] = pd.to_numeric(df[col], errors="coerce", downcast="integer")

            elif r_class == "logical":
                df[col] = df[col].map({True: True, False: False}).astype("boolean")

            # Maska po spremembi
            na_after = df[col].isna()

            # Poišči nove manjkajoče vrednosti
            newly_na = (~na_before) & (na_after)
            if newly_na.any():
                changed_rows = newly_na[newly_na].index.tolist()
                print(f"⚠️  Stolpec '{col}': {len(changed_rows)} novih NA po pretvorbi na podlagi JSON definicije.")
                print(f"    Vrstice: {changed_rows[:10]}{' ...' if len(changed_rows) > 10 else ''}\n")

    return df


def nastavi_HFD_oznako(df):
    # Določimo ustrezne pare stolpcev
    for x in range(1, 5):
        tp_col = f"TP_{x}_Dia"
        hfd_col = f"HFD_{x}_Dia"

        # Pogoj: samo za vrstice, kjer ID se začne z "RE" in TP_X_Dia ni NaN
        mask = df['ID'].astype(str).str.startswith('RE') & df[tp_col].notna()

        # Nastavimo vrednost "Y" v ustrezni HFD_X_Dia, kjer velja pogoj
        df.loc[mask, hfd_col] = 'Y'

    return df


def stratificirani_foldi(class_labels, fraction_labels, care_labels, n_splits=10, random_state=42):
    # Združi oznake v eno "kombinirano ključno besedo"
    combo_labels = [f"{c}_{f}_{l}" for c, f, l in zip(class_labels, fraction_labels, care_labels)]

    # Ustvari DataFrame
    df = pd.DataFrame({
        'class': class_labels,
        'fraction': fraction_labels,
        'care': care_labels,
        'combo': combo_labels
    })

    # Stratified split po kombinaciji
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    df['fold'] = -1

    for fold, (_, val_idx) in enumerate(skf.split(df, df['combo'])):
        df.loc[val_idx, 'fold'] = fold

    return df

def fold_combo_stats(df):
    # Preštej število pojavljanj (count) po foldih in combo-labelih
    counts = df.groupby(['fold', 'combo']).size().unstack(fill_value=0)

    # Izračunaj deleže znotraj posameznega folda
    fractions = counts.div(counts.sum(axis=1), axis=0).round(3)

    # Vrni oba: absolutne vrednosti in relativne deleže
    return counts, fractions

def build_class_labels(df):
    # Definiramo HFD-stolpce
    hfd_cols = [f"HFD_{i}_Dia" for i in range(1, 5)]

    # Preverimo, ali je v kateri od teh stolpcev vrednost "Y"
    # .eq("Y") naredi DataFrame boolean; .any(axis=1) poišče, če je katera True
    mask = df[hfd_cols].eq("Y").any(axis=1)

    # Mapiramo True→"Y", False→"N" in vrnemo kot list
    class_labels = mask.map({True: "Y", False: "N"}).tolist()
    return class_labels


def build_fraction_labels(df):
    cols = [f"SLVF_{i}_Echo" for i in range(1, 5)] + [f"LVEF_{i}_Echo" for i in range(1, 5)]
    relevant = df[cols]

    labels = []

    for _, row in relevant.iterrows():
        if row.isna().all():
            labels.append("N")
        elif (row < 50).any():
            labels.append("R")
        else:
            labels.append("P")

    return labels

def build_care_labels(df):
    return df["ID"].apply(lambda x: "P" if str(x).startswith("UT") else "S").tolist()


def shrani_csv_folde(df_data, folds_df, output_dir="foldi"):
    os.makedirs(output_dir, exist_ok=True)

    if not 'fold' in folds_df.columns:
        raise ValueError("folds_df mora vsebovati stolpec 'fold'.")

    # Preveri, da imata oba enako dolžino
    if len(df_data) != len(folds_df):
        raise ValueError("df_data in folds_df morata imeti enako število vrstic.")

    for fold in sorted(folds_df['fold'].unique()):
        # Indeksi vrstic za testni fold
        test_idx = folds_df.index[folds_df['fold'] == fold]
        train_idx = folds_df.index[folds_df['fold'] != fold]

        # Izberi vrstice iz originalnih podatkov
        df_train = df_data.loc[train_idx]
        df_test = df_data.loc[test_idx]

        # Shrani
        df_train.to_csv(os.path.join(output_dir, f"fold_{fold}_train.csv"), index=False)
        df_test.to_csv(os.path.join(output_dir, f"fold_{fold}_test.csv"), index=False)

        print(f"✅ Fold {fold}: {len(df_train)} train, {len(df_test)} test")





#
# Zdruzevanje csv podatkov po modalitetah
#
if True:
    # Seznam CSV datotek za branje
    file_names = ['Phy', 'Dia', 'Blo', 'ECG', 'Echo', 'Med', 'MRI', 'Pat', 'Spi', 'Sym', 'Xray']

    # Začetni DataFrame (bo napolnjen z merge-anjem)
    merged_df = None

    # Preberi in združi vse datoteke po stolpcu 'ID'
    for file in file_names:
        df = pd.read_csv("data\\" + file + ".csv", encoding='windows-1252', low_memory=False)
        df = df.rename(lambda x: f"{x}_{file}" if x != 'ID' else x, axis='columns')

        if 'ID' not in df.columns:
            raise ValueError(f"Datoteka {file}.csv ne vsebuje stolpca 'ID'!")

        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='ID', how='outer')  # uporabi outer merge za ohranjanje vseh ID-jev

    merged_df.to_csv("data\\merged_data.csv", index=False, encoding='utf-8')

#
# Podatkovni tipi
#
if True:
    df = pd.read_csv("data/merged_data.csv", low_memory=False, encoding='utf-8')
    print(f"Število vrstic: {df.shape[0]}")
    print(f"Število stolpcev: {df.shape[1]}")

    json_list = [
        "data\\column_metadata_Phy.json",
        "data\\column_metadata_Dia.json",
        "data\\column_metadata_Blo.json",
        "data\\column_metadata_ECG.json",
        "data\\column_metadata_Echo.json",
        "data\\column_metadata_Med.json",
        "data\\column_metadata_MRI.json",
        "data\\column_metadata_Pat.json",
        "data\\column_metadata_Spi.json",
        "data\\column_metadata_Sym.json",
        "data\\column_metadata_Xray.json"
    ]

    df = apply_column_metadata(df, json_list)
    df.to_csv("data\\merged_data.csv", index=False, encoding='utf-8')


#
# Dodajanje diagnoz za Regensburg
#
if True:
    df = pd.read_csv("data/merged_data.csv", low_memory=False, encoding='utf-8')
    df = nastavi_HFD_oznako(df)
    df.to_csv("data\\merged_data_full.csv", index=False, encoding='utf-8')

#
# Razbitje podatkov na folde
#
if True:
    df = pd.read_csv("data/merged_data_full.csv", low_memory=False, encoding='utf-8')

    class_labels = build_class_labels(df)
    fraction_labels = build_fraction_labels(df)
    care_labels = build_care_labels(df)

    folds_df = stratificirani_foldi(class_labels, fraction_labels, care_labels, n_splits=10)
    # Dobimo statistiko po foldih
    counts, fractions = fold_combo_stats(folds_df)

    print("Absolutne vrednosti po foldih:")
    print(counts)

    print("\nRelativni deleži po foldih:")
    print(fractions)

    shrani_csv_folde(df, folds_df, output_dir="data/CV_folds")



