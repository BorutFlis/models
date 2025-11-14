from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold, KFold

from abstract_models.data_source import DataSource


@dataclass
class PatientLevelDataSource(DataSource):
    _data: pd.DataFrame
    dataset_name: str = "patient_level"
    target: str = "death_event"
    cv_n_fold: int = 5
    stratification = True
    group_col = None

    def xy(self):
        X = self._data.drop(self.target, axis=1)
        X = X.loc[:, ["gender", "days_in_observation", "Phy_Age"]]
        y = [self._data[self.target]]
        return (X, y)

    def train_test(self):
        X, y = self.xy()
        return train_test_split(X, y)

    def get_cv_split_method(self, n_folds_arg=None):
        n_folds = self.cv_n_fold if n_folds_arg is None else n_folds_arg
        cv_method = KFold(n_splits=n_folds)
        return cv_method.split


@dataclass
class ClassificationDPDataSource(DataSource):
    _data: pd.DataFrame
    dataset_name: str = "classifcation_dp"
    target: str = "death_event"
    cv_n_fold: int = 5
    stratification = True
    group_col = None

    def xy(self):
        X = self._data.drop(self.target, axis=1)
        y = self._data[self.target]
        return (X, y)

    def train_test(self):
        X, y = self.xy()
        return train_test_split(X, y)

    def get_cv_split_method(self, n_folds_arg=None):
        n_folds = self.cv_n_fold if n_folds_arg is None else n_folds_arg
        cv_method = KFold(n_splits=n_folds)
        return cv_method.split

