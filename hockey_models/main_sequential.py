import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import KBinsDiscretizer

# setting path
sys.path.append('..')
from abstract_models.utils import sort_stratified_regression_group_k, get_most_important_feature, get_preliminary_accuracy, outliers_per_attr

df = pd.read_csv("data/raw/predict_points.csv")

