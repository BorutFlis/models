# Hospitalization experiments

Run from the repository root in the `models` Python environment:

```powershell
python disease_progression/scripts/hospitalization/reproduce_experiments.py
```

This separate experiment preserves the existing mortality experiments. It discovers
all 17 `post_*hosp*` targets marked `targets` in the supplied attribute schema.
The original `emmergency` spelling is retained to match the dataset.
Only schema predictors are used: patient IDs, dates, mortality labels, high-risk
labels, and all future hospitalization outcomes are excluded.

Before evaluation, the `have_hosp_stats_filter` from
`weighted_samples_experiment.py` retains rows having at least one nonmissing
value across the pre/post hospitalization statistics (including time to emergency
admission). Its separate registration-history restriction and mortality sample
weighting rules are not part of this hospitalization availability filter.
Configuration records the filter columns and row counts. Target status counts
refer to this filtered cohort, or the sample drawn from it when `--max-rows` is used.

Each observed numeric outcome is evaluated using RandomForestRegressor and a
training-fold mean baseline. Five shuffled folds use seed 42; repeated patient IDs
switch to GroupKFold. Median imputation and categorical encoding reuse
`abstract_models.imputation.median_imputer` and are fitted inside each training
fold. Forests use 100 trees, depth 16, and minimum leaf size 2. No tuning is
performed. The final forest is fitted on all observed labels for that target to
produce sorted, encoded-feature impurity importances; these are descriptive and
can favor high-cardinality features.

Outputs default to `disease_progression/data/results/hospitalization`:

- `summary.csv`: pooled out-of-fold MAE, RMSE and R2 for each target and model.
- `target_status.csv`: observed/missing counts and explicit skipped-target reasons.
- `configuration.json`: settings, target/predictor lists, input hashes and sklearn version.
- Per-target `fold_metrics.csv` and `feature_importance.csv`.
- Per-target `oof_predictions.csv` for datasets under 10,000 rows, or with `--save-oof`.
  These files contain patient IDs and should remain in the local results directory.

Missing labels are excluded separately for each target, never filled with zero.
Consequently populations vary between targets and raw errors have different units.
Time-to-admission regression describes patients with observed admission times;
it does not account for censoring. No admission occurrence labels or follow-up
eligibility rules are invented. Interpretation requires checking how missing
outcomes and follow-up windows were defined upstream. Full follow-up counts and
durations may reflect unequal observation periods.

For a quick execution check across all targets:

```powershell
python disease_progression/scripts/hospitalization/reproduce_experiments.py --max-rows 2000 --trees 5 --folds 3 --output disease_progression/data/results/hospitalization_smoke
```

Sampled results validate execution only. Use a separate output directory per run;
files with matching names are replaced. Full-data runs involve 17 separate target
experiments on nearly 500,000 rows and can take substantial time. Adjust `--jobs`
to control parallel tree fitting. Other options include `--seed`, `--trees`,
`--folds`, `--data`, and `--schema`.
