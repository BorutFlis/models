from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class Metric(ABC):

    @abstractmethod
    def calculate(self):
        pass


@dataclass
class RegressionMetric(Metric):

    mse: float = float("nan")
    mae: float = float("nan")
    r2: float = float("nan")

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.mse = mean_squared_error(y_true, y_pred)
        self.mae = mean_absolute_error(y_true, y_pred)
        self.r2 = r2_score(y_true, y_pred)

    def report(self, precision=5):
        print(f"MSE: {self.mse:15.{precision}f}")
        print(f"MAE: {self.mae:15.{precision}f}")
        print(f"R2:  {self.r2:15.{precision}f}")




