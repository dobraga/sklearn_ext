import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_selection._base import SelectorMixin


from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import OneHotEncoder


class DropIdDuplicated(BaseEstimator, SelectorMixin):
    """
    Search ID add to index, and drop Duplicated columns
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        X = X.copy()
        self.id_: list[str] = []
        self.support_: list[bool] = []
        self.dropped_: list[str] = []

        for col in X.columns:

            if (X[col].value_counts() == 1).all():
                self.id_.append(col)
            else:
                column_ok = True
                for col2 in X.columns:
                    if (
                        (col2 not in self.dropped_)
                        and (col != col2)
                        and (X[col] == X[col2]).all()
                    ):
                        column_ok = False
                        self.dropped_.append(col)
                        break

                self.support_.append(column_ok)

        self.feature_names_in_ = np.array(X.columns)
        if self.id_:
            X = X.set_index(self.id_)
        self.feature_names_out_ = X.loc[:, self._get_support_mask()].columns.to_numpy()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.set_index(self.id_).loc[:, self._get_support_mask()]

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return self.feature_names_out_
