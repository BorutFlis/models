from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold

from abstract_models.data_source import DataSource


@dataclass
class EarlyDiagnosisSource(DataSource):
    _data: pd.DataFrame
    dataset_name: str = "early_diagnosis"
    target: str = "Dia_HFD"
    cv_n_fold: int = 5
    stratification = True
    group_col = "centre"
    target_map = {"Y": 1, "N": 0}

    def xy(self):
        X = self._data.drop(self.target, axis=1)
        y = self._data[self.target].map(self.target_map)
        return (X, y)

    def train_test(self):
        X, y = self.xy()
        return train_test_split(X, y)

    def get_cv_split_method(self, n_folds_arg=None, groups=None):
        if groups is None:
            groups = self._data[self.group_col]
        n_folds = groups.nunique() if groups.nunique() < self.cv_n_fold else self.cv_n_fold
        cv_method = GroupKFold(n_splits=n_folds)
        split_func = partial(cv_method.split, groups=groups)
        return split_func


@dataclass
class EarlyDiagnosisCPRDSource(DataSource):
    _data: pd.DataFrame
    dataset_name: str = "early_diagnosis_cprd"
    target: str = "Dia_HFD"
    cv_n_fold: int = 5
    stratification = True
    group_col = "ID"

    def xy(self):
        X = self._data.drop(['Dia_HFD_6M', 'Dia_HFD_12M', 'Dia_HFD_18M'], axis=1)
        y = self._data[self.target]
        return (X, y)

    def train_test(self):
        X, y = self.xy()
        return train_test_split(X, y)

    def get_cv_split_method(self, n_folds_arg=None, groups=None):
        if groups is None:
            groups = self._data[self.group_col]
        n_folds = groups.nunique() if groups.nunique() < self.cv_n_fold else self.cv_n_fold
        cv_method = GroupKFold(n_splits=n_folds)
        split_func = partial(cv_method.split, groups=groups)
        return split_func
