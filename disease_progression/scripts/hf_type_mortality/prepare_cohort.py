"""Build and audit the filtered HES mortality cohort."""

import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils import CONFIG, PROJECT, filter_cohort, resolve_source


REQUIRED = ["patid", "date", "regstartdate", "death_event", "days_to_event",
            "HF_type", "Phy_Sex", "Phy_Age"]


def write_flowchart(flow, output):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    for step, row in enumerate(flow.itertuples(index=False)):
        y = 0.88 - step * 0.23
        ax.text(0.35, y, f"{row.stage}\nN = {row.included:,}", ha="center", va="center",
                bbox=dict(boxstyle="square,pad=0.55", fc="white", ec="#25334a"))
        if step:
            ax.annotate("", xy=(0.35, y + 0.075), xytext=(0.35, y + 0.155),
                        arrowprops=dict(arrowstyle="->", color="#25334a"))
            ax.annotate(f"Excluded\nN = {row.excluded:,}", xy=(0.35, y + 0.115),
                        xytext=(0.79, y + 0.115), ha="center", va="center",
                        arrowprops=dict(arrowstyle="<-", color="#25334a", shrinkB=0))
    fig.savefig(output / "cohort_flow.pdf", bbox_inches="tight")
    plt.close(fig)


def prepare():
    output = PROJECT / CONFIG["output"]
    output.mkdir(parents=True, exist_ok=True)
    data_path = resolve_source(PROJECT / CONFIG["dataset"])
    schema_path = resolve_source(PROJECT / CONFIG["schema"])
    schema = pd.read_csv(schema_path)
    predictors = schema.loc[schema.variable_type.eq("predictor"), "attr"].tolist()
    hospitalization = [
        name for name in schema.attr if name.startswith(("pre_", "post_")) and "hosp" in name
    ]
    if any(name.startswith("post_") for name in predictors):
        raise ValueError("Schema includes follow-up information among predictors.")
    columns = list(dict.fromkeys(REQUIRED + predictors + hospitalization))
    categorical = schema.loc[schema.dtype.eq("object") & schema.attr.isin(columns), "attr"]
    data = pd.read_csv(data_path, usecols=columns, dtype=dict.fromkeys([*categorical, "patid"], str),
                       skipinitialspace=True, low_memory=False)
    for name in categorical:
        data[name] = data[name].str.strip().replace("", float("nan"))
    cohort, flow = filter_cohort(data, hospitalization)
    cohort = cohort[list(dict.fromkeys(REQUIRED + predictors))]
    provenance = dict(CONFIG, hospitalization_columns=hospitalization, predictors=predictors)
    for key, path in [("dataset", data_path), ("schema", schema_path)]:
        with path.open("rb") as source:
            provenance[f"{key}_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    cohort_path = PROJECT / CONFIG["cohort"]
    cohort.to_parquet(cohort_path, index=False)
    cohort_schema = schema.set_index("attr").reindex(cohort.columns).rename_axis("attr")
    cohort_schema.loc[["patid", "date"], ["dtype", "variable_type"]] = ["object", "meta"]
    cohort_schema.loc["days_to_event", ["dtype", "variable_type"]] = [
        str(cohort.days_to_event.dtype), "targets"
    ]
    cohort_schema["modality"] = cohort_schema.index.map(feature_modality)
    cohort_schema.to_csv(cohort_path.with_name(f"{cohort_path.stem}_schema.csv"))
    flow.to_csv(output / "cohort_flow.csv", index=False)
    write_flowchart(flow, output)
    (output / "configuration.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(flow.to_string(index=False), flush=True)
    return cohort, predictors


def feature_modality(name):
    prefix = name.split("_")[0]
    modalities = {"Phy": "physical", "Blo": "blood", "MRI": "MRI", "ECG": "ECG",
                  "Echo": "echocardiography", "Pat": "patient history",
                  "pre": "hospitalization"}
    return modalities.get(prefix, "")


if __name__ == "__main__":
    prepare()
