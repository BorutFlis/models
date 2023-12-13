from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from abstract_models.data_source import DataSource


@dataclass
class HousePriceSource(DataSource):
    _data: pd.DataFrame

    def xy(self):
        return (self._data.drop("SalePrice", axis=1), self._data["SalePrice"])

    def train_test(self):
        X, y = self.xy()
        return train_test_split(X, y)
