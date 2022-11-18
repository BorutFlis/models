import sys

# setting path
sys.path.append('..')

import pandas as pd

from abstract_models.runner import RFRegressorRunner
from data.source import HousePriceSource
from data.loader import load_house_price_data

if __name__ == "__main__":
    ds = HousePriceSource(load_house_price_data("house_price_data/train.csv"))

    rf = RFRegressorRunner(ds)
    result = rf.run()
    result.report()
