import pandas as pd


def load_data(file_location: str) -> pd.DataFrame:
    df = pd.read_csv(file_location)
    return df

