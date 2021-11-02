from sklearn_ext.datasets import load_br_covid
from sklearn_ext.feature_selection import DropIdDuplicated

import pandas as pd


def test_drop_id_duplicated_without_id():
    df = load_br_covid()
    df["epi_week2"] = df["epi_week"]
    df = df[["epi_week", "epi_week2", "country", "totalCases"]]

    drop = DropIdDuplicated().fit(df)

    assert drop._get_support_mask() == [False, True, True, True]


def test_drop_id_duplicated_with_id():
    df = load_br_covid().reset_index()
    df["epi_week2"] = df["epi_week"]
    df = df[["index", "epi_week", "epi_week2", "country", "totalCases"]]

    drop = DropIdDuplicated().fit(df)

    assert drop.index_ == ["index"]

    assert drop._get_support_mask() == [False, True, True, True]


def test_drop_id_duplicated_with_date():
    df = load_br_covid()
    df["date"] = pd.to_datetime(df["date"])

    drop = DropIdDuplicated().fit(df)

    assert drop.index_feature_ == ["date"]
    assert drop.index_ == []
    assert list(drop.transform(df).index.names) == ["date"]
    assert "date" in list(drop.transform(df).columns)
