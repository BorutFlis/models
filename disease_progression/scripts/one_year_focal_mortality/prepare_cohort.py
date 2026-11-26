"""Build the filtered CPRD cohort, its schema, and a cohort flow diagram."""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import CONFIG, IDENTITY, OUTCOMES, PROJECT, TARGET, filter_cohort

sys.path.insert(0, str(PROJECT.parent))
from abstract_models.data_source import resolve_source


def write_flowchart(flow, output):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    for step, row in enumerate(flow.itertuples(index=False)):
        y = 0.82 - step * 0.32
        ax.text(0.4, y, f"{row.stage}\nN = {row.included:,}", ha="center", va="center",
                bbox=dict(boxstyle="square,pad=0.45", fc="white", ec="black"))
        if step:
            ax.annotate("", xy=(0.4, y + 0.09), xytext=(0.4, y + 0.23),
                        arrowprops=dict(arrowstyle="->", color="black"))
            ax.annotate(f"Excluded\nN = {row.excluded:,}", xy=(0.4, y + 0.16),
                        xytext=(0.81, y + 0.16), ha="center", va="center",
                        arrowprops=dict(arrowstyle="<-", color="black"))
    fig.savefig(output / "cohort_flow.pdf", bbox_inches="tight")
    plt.close(fig)


def prepare():
    schema = pd.read_csv(resolve_source(PROJECT / CONFIG["schema"]))
    predictors = schema.loc[schema.variable_type.eq("predictor"), "attr"].tolist()
    required = [*IDENTITY, "regstartdate", *OUTCOMES]
    if set(predictors).intersection(required) or any(
        name.startswith(("post_", "death_", "high_risk")) for name in predictors
    ):
        raise ValueError("Predictor schema contains identifiers or outcome information.")
    hospitalization = [
        name for name in schema.attr if name.startswith(("pre_", "post_")) and "hosp" in name
    ]
    columns = list(dict.fromkeys([*required, *predictors, *hospitalization]))
    categorical = schema.loc[schema.dtype.eq("object") & schema.attr.isin(columns), "attr"]
    data = pd.read_csv(
        resolve_source(PROJECT / CONFIG["dataset"]), usecols=columns,
        dtype=dict.fromkeys([*categorical, *IDENTITY, "regstartdate"], str),
        skipinitialspace=True, low_memory=False,
    )
    for name in categorical:
        data[name] = data[name].str.strip().replace("", np.nan)
    cohort, flow = filter_cohort(data, hospitalization)
    cohort = cohort[[*required, *predictors]]
    duration = cohort["days_to_event"]
    valid = np.isfinite(duration) & duration.ge(0) & cohort["death_event"].isin([0, 1])
    past_horizon = duration.gt(CONFIG["horizon_days"])
    cohort[TARGET] = np.nan
    cohort.loc[valid & past_horizon, TARGET] = 0
    cohort.loc[valid & ~past_horizon & cohort["death_event"].eq(1), TARGET] = 1

    cohort_path = PROJECT / CONFIG["cohort"]
    cohort.to_parquet(cohort_path, index=False)
    cohort_schema = schema.set_index("attr").reindex(cohort.columns).rename_axis("attr")
    cohort_schema["dtype"] = cohort.dtypes.astype(str)
    cohort_schema.loc[required[:3], "variable_type"] = "meta"
    cohort_schema.loc[[*OUTCOMES, TARGET], "variable_type"] = "targets"
    cohort_schema["modality"] = ""
    modalities = {
        "Phy": "physical", "Blo": "blood", "MRI": "MRI", "ECG": "ECG",
        "Echo": "echocardiography", "Pat": "patient history", "pre": "hospitalization",
        "Sym": "symptoms", "HF": "heart failure phenotype",
    }
    cohort_schema.loc[predictors, "modality"] = [
        modalities.get(name.split("_")[0], "") for name in predictors
    ]
    cohort_schema.to_csv(cohort_path.with_name(f"{cohort_path.stem}_schema.csv"))

    output = PROJECT / CONFIG["output"]
    output.mkdir(parents=True, exist_ok=True)
    flow.to_csv(output / "cohort_flow.csv", index=False)
    write_flowchart(flow, output)
    print(flow.to_string(index=False), flush=True)


if __name__ == "__main__":
    prepare()
