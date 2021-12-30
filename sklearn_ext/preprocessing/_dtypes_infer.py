from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Callable, Dict
import pandas as pd
import numpy as np


def to_datetime(x: pd.Series) -> pd.Series:
    return pd.to_datetime(x, infer_datetime_format=True, utc=False)


def to_numeric(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x.str.replace(",", "."))


class DtypesInfer(BaseEstimator, TransformerMixin):
    """
    Infer and transform dtypes [datetime|numeric] in dataset
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        self.dtypes_: Dict[str, Callable] = {}
        self.feature_names_in_: np.ndarray = np.array(X.columns)

        for col in X.columns:
            if X[col].dtype == "object":
                for t in [to_datetime, to_numeric]:
                    try:
                        _ = t(X[col])
                        self.dtypes_[col] = t
                        break
                    except:
                        pass

        return self

    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        check_is_fitted(self)
        X = X.copy()

        for col, transformer in self.dtypes_.items():
            X[col] = transformer(X[col])

        return X

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return self.feature_names_in_
