import os
import json

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold
from lightgbm import LGBMClassifier
from scipy.special import expit
from sklearn.utils.validation import check_is_fitted

from abstract_models.imputation import median_imputer
from abstract_models.param_grid import lgb_param_grid
from abstract_models.experiment_utils import run_cross_validation, run_cross_validation_wei
from abstract_models.metric_utils import compute_binary_classification_metrics_adjusted


class CustomBCEClassifier(LGBMClassifier):

    @staticmethod
    def _bce_objective(y_true, raw_pred):
        """
        Binary cross-entropy objective.

        raw_pred contains logits, NOT probabilities.
        """
        p = expit(raw_pred)

        grad = p - y_true
        hess = p * (1.0 - p)

        return grad, hess

    @staticmethod
    def _weighted_bce(y_true, raw_pred, weight):
        p = expit(raw_pred)

        grad = weight * (p - y_true)
        hess = weight * p * (1 - p)

        return grad, hess

    @staticmethod
    def _focal_loss(y_true, raw_pred, alpha=0.25, gamma=5.0):
        """
        Binary focal loss for LightGBM.

        Parameters
        ----------
        y_true : array-like
            Binary labels: 0 or 1.

        raw_pred : array-like
            LightGBM raw predictions (logits).

        alpha : float
            Weight assigned to the positive class.
            Negative class gets (1 - alpha).

        gamma : float
            Focusing parameter.
            gamma=0 reduces to alpha-weighted BCE.

        Returns
        -------
        grad : ndarray
            First derivative w.r.t. raw_pred.

        hess : ndarray
            Second derivative w.r.t. raw_pred.
        """

        y = np.asarray(y_true)
        z = np.asarray(raw_pred)

        p = expit(z)

        # p_t = probability assigned to the correct class
        pt = np.where(y == 1, p, 1.0 - p)

        # alpha_t = class weighting
        at = np.where(y == 1, alpha, 1.0 - alpha)

        # Numerical stability
        eps = 1e-12
        pt = np.clip(pt, eps, 1.0 - eps)

        # Convenient sign:
        # y=1 -> +1
        # y=0 -> -1
        s = 2.0 * y - 1.0

        # d(pt)/dz
        dpt = s * p * (1.0 - p)

        # d²(pt)/dz²
        d2pt = s * p * (1.0 - p) * (1.0 - 2.0 * p)

        # Focal loss as a function of pt:
        #
        # L(pt) = -at * (1-pt)^gamma * log(pt)

        # dL/dpt
        dL_dpt = at * (
            gamma * (1.0 - pt) ** (gamma - 1.0) * np.log(pt)
            - (1.0 - pt) ** gamma / pt
        )

        # d²L/dpt²
        d2L_dpt2 = at * (
            -gamma * (gamma - 1.0)
            * (1.0 - pt) ** (gamma - 2.0)
            * np.log(pt)
            + 2.0 * gamma
            * (1.0 - pt) ** (gamma - 1.0) / pt
            + (1.0 - pt) ** gamma / (pt ** 2)
        )

        # Chain rule
        grad = dL_dpt * dpt

        hess = (
            d2L_dpt2 * dpt**2
            + dL_dpt * d2pt
        )

        return grad, hess

    def __init__(self, threshold=0.5, **kwargs):
        self.threshold = threshold

        # Our objective is fixed to custom BCE
        kwargs.pop("objective", None)

        super().__init__(
            objective=self._focal_loss,
            **kwargs
        )

    def _predict_raw(self, X, **kwargs):
        """Get raw LightGBM logits without classifier post-processing."""

        check_is_fitted(self)

        return self.booster_.predict(
            X,
            raw_score=True,
            **kwargs
        )

    def predict_proba(self, X, **kwargs):
        """
        Return:
            column 0 -> P(y=0)
            column 1 -> P(y=1)
        """

        raw_pred = self._predict_raw(X, **kwargs)

        p1 = expit(raw_pred)

        return np.column_stack([
            1.0 - p1,
            p1
        ])

    def predict(self, X, **kwargs):
        """Return binary class labels."""

        p1 = self.predict_proba(X, **kwargs)[:, 1]

        return (p1 >= self.threshold).astype(int)


DATA_DIR = "../data"
DATA_DUMP_DIR = "../data_dump"
RESULTS_DIR = os.path.join(DATA_DIR, "results")

attr_selections = json.load(open(os.path.join(DATA_DIR, "expert_attr_selection.json")))

df = pd.read_csv(os.path.join(DATA_DIR, "processed", "balanced_ED_NT.csv"), index_col=[0, 1])
attrs = list(set(attr_selections["expert"]).intersection(df.columns)) + ['Med_LD_permanent']

w = pd.Series(index=df.index)
w.loc[df["Blo_NT"].gt(125) & df["Dia_HFD_12M"].eq(0)] = df.loc[df["Blo_NT"].gt(125) & df["Dia_HFD_12M"].eq(0), "Blo_NT"] - 125
w = w.fillna(0.5)

X = df.loc[:, attrs]
y = df["Dia_HFD_12M"]

model = CustomBCEClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)

pipeline = Pipeline(
    steps=[('preprocessor', median_imputer), ('classifier', model)]
)

base_model_process = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=lgb_param_grid,
    cv=None,
    scoring='accuracy',
    n_jobs=10,
    verbose=1,
    n_iter=3,
)

cv_method = KFold(n_splits=5, shuffle=True)

split_func = cv_method.split

cv_results = run_cross_validation(
    split_func, pipeline, compute_binary_classification_metrics_adjusted, X, y
)

# cv_results.to_csv(os.path.join(DATA_DUMP_DIR, "1_minus_p_hess_BCE.csv"), index=None)