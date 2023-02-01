from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import plot_partial_dependence


df = pd.read_csv("house_price_data/train.csv")
df = df.drop("Id", axis=1)
y = df.pop("SalePrice")
df = df.select_dtypes(include=["float64", "int64"]).dropna(axis=1)

rf = RandomForestRegressor()
rf.fit(df.values, y.values)

fig,ax = plt.subplots(figsize=(12, 4))
plot_partial_dependence(rf, df.values, df.columns[np.argsort(rf.feature_importances_)[-3:]], grid_resolution=20, ax=ax)