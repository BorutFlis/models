def transform_prospective_df(pro_df):
    prospective_map = {
        "v1_sex": "Phy_Sex", "age": "Phy_Age", "myocardial_infarction": "Pat_PMI", "cabg": "Pat_CABG",
        "shortness_of_breath": "Sym_DAR", "atrial_fibrillation2": "Pat_AF", "diabetes_mellitus_type_2": "Pat_DM2",
        "diabetes_mellitus_type_1": "Pat_DM",
        "v1_weight": "Phy_Wei", "v1_height": "Phy_Hei", "ntprobnp_value": "Blo_NT", "creatinine_value": "Blo_Cre",
        "hypertension": "Pat_Hyp", "chest_pain2": "Sym_CPR"
    }

    pro_df = pro_df.rename(columns=prospective_map)
    pro_df.loc[pro_df["Pat_DM2"].eq(1) | pro_df["Pat_DM"].eq(1), "Pat_DM"] = "Y"
    pro_df.loc[~pro_df["Pat_DM"].eq("Y"), "Pat_DM"] = "N"
    pro_df["Phy_Sex"].map({1: "Male", 2: "Female"})
    pro_df["Phy_Sex"] = pro_df["Phy_Sex"].map({1: "Male", 2: "Female"})
    pro_df["Sym_DAR"] = pro_df["Sym_DAR"].map({0: "N", 1: "Y"})
    pro_df["Pat_CABG"] = pro_df["Pat_CABG"].map({0: "N", 1: "Y"})
    pro_df["Sym_CPR"] = pro_df["Sym_CPR"].map({0: "N", 1: "Y"})
    pro_df["Pat_PMI"] = pro_df["Pat_PMI"].map({0: "N", 1: "Y"})
    pro_df["Pat_Hyp"] = pro_df["Pat_Hyp"].map({0: "N", 1: "Y"})
    pro_df["Pat_AF"] = pro_df["Pat_AF"].map({0: "N", 1: "Y"})
    return pro_df.loc[:, prospective_map.values()]



