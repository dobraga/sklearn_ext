from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)


class DatetimeEncoder(TransformerMixin, BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        self.feature_names_in_ = np.array(X.columns)
        self.feature_names_out_ = np.empty(0, dtype="object")

        for col in self.feature_names_in_:
            self.feature_names_out_ = np.append(
                self.feature_names_out_,
                [
                    f"{col}_year",
                    f"{col}_week",
                    f"{col}_day",
                    f"{col}_month",
                    f"{col}_quarter",
                ],
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        new_X = np.zeros((X.shape[0], 0))

        for col in self.feature_names_in_:
            aux = (
                X[col]
                .dt.isocalendar()
                .assign(month=X[col].dt.month, quarter=X[col].dt.quarter)
                .astype("int16")
            )

            new_X = np.concatenate((new_X, aux), axis=1)

        return pd.DataFrame(new_X, columns=self.get_feature_names_out(), index=X.index)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self)
        return self.feature_names_out_
