from abc import ABC, abstractmethod


class DataSource(ABC):

    @abstractmethod
    def xy(self):
        pass

    @abstractmethod
    def train_test(self):
        pass
