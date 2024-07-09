import pandas as pd
import numpy as np
from pandera import Column, DataFrameModel, Check, Field, check
from pandera.typing import DateTime, String, Series, Float, Int, Bool, Category


class HousePrice(DataFrameModel):
    class Config:
        coerce = True

    CentralAir: Series[Category] = Field(isin=["Y", "N"], nullable=True)