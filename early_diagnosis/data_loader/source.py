from dataclasses import dataclass

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

    def xy(self):
        X = self._data.drop(self.target, axis=1)
        y = self._data[self.target].map({"Y": 1, "N": 0})
        return (X, y)

    def train_test(self):
        X, y = self.xy()
        return train_test_split(X, y)

    def get_cv_split_method(self, n_folds_arg=None):
        n_folds = n_folds_arg if n_folds_arg is not None else self.cv_n_fold

        return GroupKFold(n_splits=n_folds)