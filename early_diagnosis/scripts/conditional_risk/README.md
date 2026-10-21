# Conditional heart failure risk experiment

Run from the repository root using the models conda environment:

```powershell
python -m early_diagnosis.scripts.conditional_risk.reproduce_experiments
```

Samples **200 eligible patients per file** from `early_diagnosis/data/raw/cprd_NT/29.csv`
and `8.csv` using seed 42, retaining **every observed pre-diagnosis timepoint**
for the sampled patients. An NT measurement is not required at each timepoint.
Each timepoint becomes a new prediction baseline, with its own future event and
follow-up times. Diagnosis-day and post-diagnosis rows are excluded because this
experiment predicts incident HF. Clinical predictors are forward filled
within each patient's history, without using later measurements. Histories with
overlapping mapped patient IDs across files cause an error. Diagnosed patients
with no usable event date are excluded and counted.

Three separate RandomForestClassifier pipelines estimate conditional HF risk in
(0, 1], (1, 2], and (2, 5] years from baseline. An interval's training set contains
only patients HF-free at its start whose outcome is known: HF during the interval,
a later HF event, or event-free observation through its end. Patients censored
inside an interval without an event are excluded from that interval's training.
An event exactly at year 1 belongs to the first interval; an event exactly at
year 2 belongs to the second. A year is 365.25 days.

For conditional probabilities h1, h2, h3, cumulative risks are
`1 - cumprod(1 - [h1, h2, h3])`. The 1-, 2-, and 5-year probabilities are therefore
nondecreasing for every patient. The third conditional probability covers three
years; it is not an annual hazard. All predictors remain baseline measurements.

Validation uses five shared patient folds stratified by eventual diagnosis.
Every timepoint of a patient stays in the same fold, preventing within-patient
leakage. Each timepoint receives three conditional predictions from models trained
entirely outside its patient's fold, even if its outcomes are unknown. Preprocessing is
fitted inside each training fold and interval, using cloned
`abstract_models.imputation.median_imputer` pipelines with constant missing
categories and retained empty features. Cumulative metrics use only known
horizon outcomes; conditional metrics also exclude patients no longer at risk.
The experiment reuses `abstract_models` binary metrics and supplements them with
average precision and Brier score. The hard classification threshold is 0.5.

Training and metrics weight each eligible timepoint equally; longer patient
histories therefore contribute more rows. Positive timepoint counts are not
counts of unique HF diagnoses. Both patient and timepoint counts are reported.

After OOF evaluation, three final pipelines are refitted using every eligible
timepoint with a known outcome for the respective interval. Feature importances
come from these full-data fits, sorted descending per classifier. Importance
names describe transformed features, including encoded categorical levels.

The script follows the supplied coding standards with a dedicated
`scripts/conditional_risk/reproduce_experiments.py` entry point, random forests,
`target_container` and `gather_` naming, a processed attribute schema and OOF
predictions for all patients (including when the cohort exceeds 10,000).
Existing unrelated experiments are preserved.

Outputs in `early_diagnosis/outputs/conditional_risk_all_timepoints/`:

- `oof_predictions.csv`: one row per patient and timepoint, fold, interval labels and
  probabilities, cumulative labels and risks. Unknown labels remain blank.
- `metrics.csv`: pooled OOF conditional and cumulative metrics.
- `fold_counts.csv`: training sample and event counts per interval and fold.
- `feature_importance.csv`: sorted full-data random forest importances.
- `conditional_models.joblib`: dictionary of three fitted pipelines.
- `cohort_counts.csv`: raw, eligible and sampled patient counts, plus retained
  timepoint counts, for each source file.
- `run_config.json`: parameters, feature names, targets and validation metadata.

`early_diagnosis/data/processed/conditional_risk_timepoints.csv` contains the
timepoint cohort and targets. `conditional_risk_timepoints_schema.csv` identifies each
attribute's dtype and role (`predictor`, `targets`, or `meta`). Source identifiers,
dates, follow-up and original diagnosis columns are metadata, never predictors.

For inference, supply baseline features with the same preparation and column
names as the processed cohort:

```python
import joblib
import pandas as pd
from early_diagnosis.scripts.conditional_risk.reproduce_experiments import predict_risks

models = joblib.load("early_diagnosis/outputs/conditional_risk_all_timepoints/conditional_models.joblib")
schema = pd.read_csv("early_diagnosis/data/processed/conditional_risk_timepoints_schema.csv")
predictors = schema.loc[schema.variable_type.eq("predictor"), "attr"].tolist()
cohort = pd.read_csv("early_diagnosis/data/processed/conditional_risk_timepoints.csv")
risks = predict_risks(models, cohort[predictors])
```

These are research estimates, without post-hoc calibration. Follow-up for
undiagnosed patients ends at the last recorded row; days_to_HFD is assumed to
identify the first HF diagnosis. Complete-interval censoring exclusions discard
partial follow-up and can introduce selection bias. Death is not modeled as a
competing event. OOF metrics assess discrimination and calibration within this
sampled cohort, not external clinical validity. Outputs retain patient IDs.

Options include `--files`, `--data-dir`, `--output-dir`, `--processed-dir`,
`--patients-per-file`, `--folds`, `--n-estimators`, `--n-jobs` and `--seed`;
see `--help`. The previous first-NT-baseline results remain in
`early_diagnosis/outputs/conditional_risk/` for comparison.
