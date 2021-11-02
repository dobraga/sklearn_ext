from sklearn_ext.datasets import load_br_covid
from sklearn_ext.feature_selection import DropIdDuplicated


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

    assert drop.id_ == ["index"]

    assert drop._get_support_mask() == [False, True, True, True]
