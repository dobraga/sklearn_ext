from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)


class DatetimeEncoder(TransformerMixin, BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        self.feature_names_in_ = np.array(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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

    def get_feature_names_out(self, input_features=None):
        features = []

        for col in self.feature_names_in_:
            features += [
                f"{col}_year",
                f"{col}_week",
                f"{col}_day",
                f"{col}_month",
                f"{col}_quarter",
            ]

        return features
