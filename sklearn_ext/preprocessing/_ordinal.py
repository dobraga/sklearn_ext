from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)


class OrdinalEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, mapping_: dict, nan_value: str = "NAN"):
        """
        dict_ordinal = {
            'BsmtCond': ['Ex','Gd','TA','Fa','Po','NAN'],
            'MSZoning': ['A','C (all)','FV','I','RH','RL','RP','RM', 'NAN'],
            'Street': ['Grvl','Pave']
        }
        """
        super().__init__()
        self.nan_value = nan_value
        self.mapping_ = mapping_

    def _transform_value(self, value: str, values: list) -> int:
        value = self.nan_value if pd.isna(value) else value
        return int(np.argwhere([value == obj for obj in values]))

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        new_X = np.zeros((X.shape[0], len(self.mapping_)))

        for i, (col, values) in enumerate(self.mapping_.items()):
            try:
                new_X[:, i] = X[col].apply(lambda v: self._transform_value(v, values))
            except Exception as e:
                raise Exception(
                    f"The column {col} was not converted, verify dict. [{e}]"
                )

        return pd.DataFrame(new_X, columns=self.get_feature_names_out(), index=X.index)

    def get_feature_names_out(self, input_features=None):
        return list(self.mapping_.keys())
