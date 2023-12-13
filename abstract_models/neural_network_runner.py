from abc import ABC, abstractmethod

import numpy as np

from .metrics import RegressionMetric, Metric
from .data_source import DataSource

import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout


class LSTMRunner(Runner):

    def __init__(self, ds: DataSource):
        self.X_train, self.X_test, self.y_train, self.y_test = ds.train_test()
        self.metric = RegressionMetric()

    def run(self) -> Metric:
        self.regressor = Sequential()
        self.regressor.add(LSTM(units=50,
                                batch_input_shape=(64, self.X_train.shape[1], 1),
                                stateful=True))
        self.regressor.add(Dropout(0.2))
        self.regressor.add(Dense(units=1, activation="sigmoid"))
        self.regressor.compile(optimizer="adam", loss="mean_squared_error")
        self.regressor.fit(self.X_train, self.y_train, epochs=20, batch_size=64)

        y_pred = self.regressor.predict(self.X_test)
        self.metric.calculate(self.y_test, y_pred)
        return self.metric

    def endpoint(self, X: np.array) -> np.array:
        return self.regressor.predict(X)
