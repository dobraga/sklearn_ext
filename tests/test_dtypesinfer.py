from sklearn_ext import datasets, preprocessing

import pandas as pd
from numpy import dtype


def test_dtypes_infer():
    df = datasets.load_br_covid()

    dinfer = preprocessing.DtypesInfer()
    dinfer.fit(df)
    assert (
        dinfer.transform(df).dtypes
        == pd.Series(
            [
                dtype("int64"),
                dtype("<M8[ns]"),
                dtype("O"),
                dtype("O"),
                dtype("O"),
                dtype("int64"),
                dtype("int64"),
                dtype("int64"),
                dtype("int64"),
                dtype("int64"),
                dtype("int64"),
                dtype("float64"),
                dtype("float64"),
                dtype("float64"),
                dtype("float64"),
                dtype("float64"),
                dtype("float64"),
                dtype("float64"),
                dtype("float64"),
                dtype("float64"),
                dtype("float64"),
                dtype("float64"),
                dtype("float64"),
                dtype("float64"),
                dtype("float64"),
                dtype("float64"),
            ],
            index=[
                "epi_week",
                "date",
                "country",
                "state",
                "city",
                "newDeaths",
                "deaths",
                "newCases",
                "totalCases",
                "deathsMS",
                "totalCasesMS",
                "deaths_per_100k_inhabitants",
                "totalCases_per_100k_inhabitants",
                "deaths_by_totalCases",
                "recovered",
                "suspects",
                "tests",
                "tests_per_100k_inhabitants",
                "vaccinated",
                "vaccinated_per_100_inhabitants",
                "vaccinated_second",
                "vaccinated_second_per_100_inhabitants",
                "vaccinated_single",
                "vaccinated_single_per_100_inhabitants",
                "vaccinated_third",
                "vaccinated_third_per_100_inhabitants",
            ],
        )
    ).all()
