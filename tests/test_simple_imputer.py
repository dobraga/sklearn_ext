from sklearn_ext.preprocessing import SimpleImputer
from sklearn_ext.datasets import load_house_prices


def test_simple_imputer():
    df, _ = load_house_prices()

    assert df.isna().sum().sum() > 0

    imputer = SimpleImputer().fit(df)

    df_imputed = imputer.transform(df)

    assert df_imputed.isna().sum().sum() == 0
