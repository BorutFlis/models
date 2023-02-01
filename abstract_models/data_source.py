from abc import ABC, abstractmethod


class DataSource(ABC):

    @abstractmethod
    def train_test(self):
        pass
