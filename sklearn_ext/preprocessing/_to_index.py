import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class ToIndex(BaseEstimator, TransformerMixin):
    """
    Search ID and add to index
    """

    def __init__(
        self, cols_index=[], cols_index_features=[], add_datetime_index=True
    ) -> None:
        super().__init__()
        self.cols_index = cols_index
        self.cols_index_features = cols_index_features
        self.add_datetime_index = add_datetime_index

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        X = X.copy()
        self.index_: list[str] = []
        self.index_feature_: list[str] = []

        for col in X.columns:
            if col in self.cols_index_features or (
                self.add_datetime_index and hasattr(X[col], "dt")
            ):
                self.index_feature_.append(col)

            elif col in self.cols_index:
                self.index_.append(col)

            elif (X[col].value_counts() == 1).all():
                self.index_.append(col)

        self.feature_names_in_ = np.array(X.columns)
        self.feature_names_out_ = self.transform(X).columns.to_numpy()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if self.index_:
            X = X.set_index(self.index_)
        if self.index_feature_:
            X = X.set_index(
                self.index_feature_, drop=False, append=len(self.index_) > 0
            )

        return X

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return self.feature_names_out_
