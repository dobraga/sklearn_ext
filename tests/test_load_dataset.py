from sklearn_ext import datasets


def test_load_dataset():
    df = datasets.load_br_covid()

    assert df.shape[1] == 26
