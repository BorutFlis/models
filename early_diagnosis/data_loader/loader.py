import pandas as pd


def load_data(file_location: str):
    df = pd.read_csv(file_location, index_col=[0, 1])
    df = pd.concat([df.index.to_frame(), df], axis=1)
    return df
