from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np


class TransformOthers(BaseEstimator, TransformerMixin):
    """
    This class will be used to transform rare value to {others_value}

    :threshold: Define threshold to transform the value others_value
    :others_value: Value
    """

    def __init__(self, threshold: float = 0.1, others_value="Others"):
        super().__init__()
        self.threshold = threshold
        self.others_value = others_value

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        self.relevant_values_ = {}

        for col in X.select_dtypes(include=["object"]):
            aux = X[col].value_counts(normalize=True, dropna=False)
            aux = aux >= self.threshold

            if any(aux):
                self.relevant_values_[col] = list(aux[aux].index)

        self.feature_names_in_ = np.array(list(self.relevant_values_.keys()))

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        new_X = np.empty((X.shape[0], len(self.feature_names_in_)), dtype="object")

        for i, (col, relevant_values) in enumerate(self.relevant_values_.items()):
            new_X[:, i] = X[col].values
            new_X[~X[col].isin(relevant_values), i] = "Others"

        return pd.DataFrame(new_X, columns=self.get_feature_names_out(), index=X.index)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_
