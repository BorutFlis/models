from abc import ABC, abstractmethod

from .metrics import RegressionMetric, Metric
from .data_source import DataSource

import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


class Runner(ABC):

    @abstractmethod
    def run(self):
        pass


class RFRegressorRunner(Runner):

    def __init__(self, ds: DataSource):
        X, y = ds.xy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X.values, y)
        self.metric = RegressionMetric()

    def run(self) -> Metric:
        rf = RandomForestRegressor()
        rf.fit(self.X_train, self.y_train)
        y_pred = rf.predict(self.X_test)
        self.metric.calculate(self.y_test, y_pred)
        return self.metric
