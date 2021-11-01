from sklearn_ext.preprocessing import DatetimeEncoder
from sklearn_ext.datasets import load_br_covid

import pandas as pd


def test_datetimeencoder():
    df = load_br_covid()

    df["date"] = pd.to_datetime(df["date"])

    dt_encoder = DatetimeEncoder().fit(df[["date"]])

    assert dt_encoder.transform(df).shape[1] == 5
