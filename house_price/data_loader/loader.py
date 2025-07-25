import pandas as pd


def load_data(file_location: str):
    df = pd.read_csv(file_location, index_col=0)
    df = df.select_dtypes(include=["float64", "int64"])
    return df
