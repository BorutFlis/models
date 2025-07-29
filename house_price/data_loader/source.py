from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from abstract_models.data_source import DataSource


@dataclass
class RealDataSource(DataSource):
    _data: pd.DataFrame
    dataset_name: str = "house_price"
    cv_n_fold: int = 5
    stratification = True
    multi_class = True

    def xy(self):
        X = self._data.drop("SalePrice", axis=1)
        y = self._data["SalePrice"]
        if self.stratification:
            if self.multi_class:
                y = [
                    y.gt(y.quantile(0.25)).astype(int)._set_name("SalePrice25"),
                    y.gt(y.quantile(0.5)).astype(int)._set_name("SalePrice50"),
                    y.gt(y.quantile(0.75)).astype(int)._set_name("SalePrice75")
                ]
            else:
                y = [y.gt(y.quantile(0.5)).astype(int)._set_name("SalePrice50")]
        return (X, y)

    def train_test(self):
        X, y = self.xy()
        return train_test_split(X, y)

    def get_cv_split_method(self):
        return KFold(n_splits=self.cv_n_fold)
