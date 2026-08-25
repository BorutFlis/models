import os
from functools import partial

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold, GroupKFold

from abstract_models.param_grid import rf_param_grid
from abstract_models.experiment_utils import run_cross_validation
from abstract_models.imputation import median_imputer
from abstract_models.metric_utils import compute_binary_classification_metrics_adjusted, mean_std_metrics_output


DATA_DIR = "../data"

experiments_to_run = ["attr_selection"]

attr_selection = [
    "Sym_DAR__ever", "Sym_DAR__in_last_year", "Sym_DAR__count", "Pat_DM__ever", "Pat_DM__in_last_year", "Pat_DM__count",
    "Pat_Hyp__ever", "Pat_Hyp__in_last_year", "Pat_Hyp__count", "Phy_Sex", "Phy_Age", 'Phy_Sys__median_1.5Y', 'Phy_Sys__last',
    'Phy_Dia__last', 'Phy_Dia__median_1.5Y', 'Blo_Hb__median_1.5Y', 'Blo_Hb__last', 'Blo_Cre__median_1.5Y', 'Blo_Cre__last'
]

if "attr_selection" in experiments_to_run:
    df = pd.read_csv(os.path.join(DATA_DIR, "processed", "high_risk_HES_big.csv"))
    df = df.dropna(subset="high_risk_1000_adjusted_no_hosp")


    X = df.loc[:, attr_selection]
    y = df["high_risk_1000_adjusted_no_hosp"].astype(int)
    pipeline = Pipeline(
        steps=[('preprocessor', median_imputer), ('classifier', RandomForestClassifier())]
    )

    base_model_process = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=rf_param_grid,
        cv=None,
        scoring='accuracy',
        n_jobs=10,
        verbose=1,
        n_iter=5,
    )
    k_fold = KFold(n_splits=10, shuffle=True)
    normal_cv_results = run_cross_validation(
        k_fold.split, base_model_process, compute_binary_classification_metrics_adjusted, X, y
    )


