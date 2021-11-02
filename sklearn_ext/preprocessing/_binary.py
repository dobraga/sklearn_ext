from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np


class TransformBinary(BaseEstimator, TransformerMixin):
    """
    Create binarized columns using a most relevant value
    """

    def __init__(self, threshold_min=0.75, threshold_max=0.95, fillna="None"):
        super().__init__()
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.fillna = fillna

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        self.feature_names_out_ = np.empty(0, dtype="object")
        self.cols_drop_ = np.empty(0, dtype="object")
        self._to_bin_ = {}

        for col in X.select_dtypes(include="object").columns:
            value_counts = X[col].fillna(self.fillna).value_counts(normalize=True)
            most_freq = value_counts.index[0]
            value = value_counts[most_freq]

            if value >= self.threshold_max:
                self.cols_drop_ = np.append(self.cols_drop_, col)

            elif self.threshold_min < value < self.threshold_max:
                self._to_bin_[col] = most_freq
                self.feature_names_out_ = np.append(
                    self.feature_names_out_, "bin_" + col + "_" + str(most_freq)
                )

        self.feature_names_in_ = np.array(list(self._to_bin_.keys()))

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        new_X = np.empty((X.shape[0], len(self._to_bin_)), dtype=object)

        for i, (col, most_freq) in enumerate(self._to_bin_.items()):
            new_X[:, i] = X[col].apply(lambda x: 1 if x == most_freq else 0)

        return pd.DataFrame(new_X, columns=self.get_feature_names_out(), index=X.index)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self)
        return self.feature_names_out_
