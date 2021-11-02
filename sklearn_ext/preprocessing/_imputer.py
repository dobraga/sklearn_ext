import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin


class SimpleImputer(BaseEstimator, TransformerMixin):
    """
    Simple input non observed data
    """

    def __init__(self, mapping: dict = {"object": "None", "number": -1.0}):
        super().__init__()
        self.mapping = mapping

    def fit(self, X: pd.DataFrame, y: np.ndarray = None):
        self.mapping_ = {
            k: (v, list(X.select_dtypes(include=k).columns))
            for k, v in self.mapping.items()
            if X.select_dtypes(include=k).shape[1] > 0
        }
        return self

    def transform(self, X: pd.DataFrame, y: np.ndarray = None) -> pd.DataFrame:
        check_is_fitted(self)
        X = X.copy()
        for _, (fillna, columns) in self.mapping_.items():
            X[columns] = X[columns].fillna(fillna)

        return X

    def get_feature_names_out(self, input_features=None):
        names = []

        for _, (_, columns) in self.mapping_.items():
            names += columns

        return names
