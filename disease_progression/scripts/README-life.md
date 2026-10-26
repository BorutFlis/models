# LIFE-Preserved assessments

Install `requirements-life.txt` in the project Python environment. Chrome must
be installed; Selenium Manager resolves the matching driver on first use.

From the repository root:

```powershell
python -m pip install -r disease_progression/scripts/requirements-life.txt
python -m disease_progression.scripts.run_life_comparison --limit 5
python -m disease_progression.scripts.run_life_comparison
```

The CLI runs the accepted HFpEF cohort preparation. Numeric complete-case
selection and the existing ischemic disease and hospitalization proxies remain
in place. The derived `Previous HF hospitalization` column is now retained in
the prediction frame. The smoking proxy remains outside that frame, as in the
accepted script; a full source row passed to `prepare_profile` can use it.

`--hb-unit gdL` is the default; use `--hb-unit mmolL` if the source measurements
are in mmol/L. Observed haemoglobin is passed in the specified unit without
conversion. The fallback 14 g/dL is converted to mmol/L when needed.

`--defaults defaults.json` overrides reference values by website input ID.
`--reference-csv training_reference.csv` fits numeric means and categorical modes
from a supplied reference cohort, with the first column used as its row ID.
Explicit JSON overrides take precedence. Do not fit on the evaluation cohort
when avoiding information leakage. The default assumptions are declared in
`life_preserved.DEFAULTS`; they are illustrative reference values, not measured
population means. Missing NYHA uses I/II, heart rate uses 70 bpm, and the recent
HF hospitalization indicator uses No. Every imputed field is recorded.

Source numeric sex uses the CSV convention 0=male, 1=female and is translated
to website coding 1=male, 0=female. Inputs provided directly as website IDs
already use website coding. Pass NYHA labels such as `III` or `III or IV`,
or site-coded binary values under `dummynyha34`.

Use `--treatment SGLT2 --treatment Fine` for a treatment scenario. With no
options, the baseline scenario selects no treatments. All nine outputs include
the website's baseline and treatment values, reductions and life expectancy
benefit for the selected scenario. Some treatments depend on BMI; an unavailable
selection is reported as a row error. Multiple treatment scenarios can be run
separately and retained in the same output directory because their checkpoint
keys differ. `predictions.csv` describes the current invocation; historical
per-row JSON checkpoints remain in their hashed subdirectories.

## Functions

```python
from functools import partial
import pandas as pd
from disease_progression.scripts.life_preserved import (
    attach_identifiers, prepare_profile, predict_profile, create_driver, run_batch,
)

mapping = attach_identifiers(pd.read_csv("mapping.csv"))
profile = prepare_profile({"age": 70, "heartrate": 72}, mapping)
driver = create_driver()
try:
    result = predict_profile(driver, profile, "results/single")
    batch = run_batch(frame, mapping, partial(predict_profile, driver), "results/batch")
finally:
    driver.quit()
```

`run_batch` accepts an injected prediction callable for alternate transports or
tests. Rows require unique IDs. Successful unchanged rows resume from checkpoints;
failed or changed rows run again. Configure `--delay`, `--retries`, `--headed`,
`--no-resume`, `--output-dir` and `--data-dir` as needed. Out-of-range values fail
without being clipped. Errors return a nonzero CLI exit code after processing.

## Results and verification

Each successful row has a `result.json`, rendered `assessment.html`, and PNGs
for `resultBarChart`, `resultBarChartLife` and `resultPolygon`. The JSON retains
all nine numeric values with their original labels, raw text and units. Percentage
values are on the 0-100 scale. Life expectancy is preserved exactly as labelled
by the app; it is not automatically converted to remaining years. The flattened
`predictions.csv` joins results to local row IDs and includes error rows.
Reduction signs are also preserved as reported by the app, which can return a
negative difference for a beneficial treatment.

Each assessment loads a fresh page, dismisses the informational modal, explicitly
sets inputs and treatment choices, observes a new calculation response, then
activates all three tabs. This supports identical consecutive predictions without
assuming that a result must change. Only predictor inputs are submitted; IDs and
other source columns stay local. Checkpoints, HTML and plots contain health data
and should be stored under the same access controls as the source dataset.

Run offline tests with:

```powershell
python -m unittest disease_progression.scripts.test_life_preserved -v
```

Verification on 25 October 2026: nine offline tests passed. Two identical
synthetic baseline profiles and one synthetic SGLT2 treatment profile each
returned all nine values and all three plots through Selenium. No patient batch
was submitted during implementation.

The public app describes its results as preliminary and not intended for clinical
use. This integration supports research comparisons. OOF comparison is not part
of this change.
