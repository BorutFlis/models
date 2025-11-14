import pandas as pd


def load_data(file_location: str) -> pd.DataFrame:
    df = pd.read_csv(file_location, index_col=0)
    return df

