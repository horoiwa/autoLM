import pandas as pd
import numpy as np

from feature_engineering.util import (onehot_conversion, poly_generation,
                                      standard_scaler, simple_mapping)


class DataSet():
    def __init__(self, criterio=15, poly=1, stsc=True):
        self.X = None
        self.y = None
        self.models = {}

        self.fmap = None
        self.cmap = None

        # temp dataset
        self.X_pre = None
        self.X_poly = None
        self.X_sc = None

        # Dataset after feature selection
        self.X_fin = None
        self.selected_features = None

        # pipeline config
        self.criterio = criterio
        self.poly = poly
        self.stsc = stsc

    def __repr__(self):
        return "DataSet Object"

    def input_check(self):
        assert self.X.shape[0] == self.y.shape[0], "Error #1"
        assert len(self.X.columns) > 0, "Error #2"
        assert len(self.y.columns) > 0, "Error #3"

    def transform(self, X):
        if type(X) == pd.Series:
            X = pd.DataFrame(X).T
        else:
            assert type(X) == pd.DataFrame

        assert np.all(X.columns.values == self.X.columns.values), "Erro #11"

        X_num = X[self.fmap["numeric"]]
        X_ord = X[self.fmap["ordinal"]]
        X_cat = X[self.fmap["category"]]

        X_onehot = onehot_conversion(X_cat, model=self.models["onehot"])
        X_pre = pd.concat([X_num, X_ord, X_onehot], 1)

        if self.poly:
            X_poly = poly_generation(X_pre, model=self.models["poly"])
        else:
            X_poly = X_pre

        if self.stsc:
            X_sc = standard_scaler(X_poly, model=self.models['stsc'])
        else:
            X_sc = X_poly

        return X_sc

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.input_check()

        self._preprocess()
        self._postprocess()

    def _preprocess(self):
        self.fmap = simple_mapping(self.X, criterio=self.criterio)

        X_num = self.X[self.fmap["numeric"]]
        X_ord = self.X[self.fmap["ordinal"]]
        X_cat = self.X[self.fmap["category"]]

        X_onehot, self.models["onehot"], self.cmap = onehot_conversion(X_cat)
        self.X_pre = pd.concat([X_num, X_ord, X_onehot], 1)

    def _postprocess(self):
        if self.poly:
            self.X_poly, self.models["poly"] = poly_generation(self.X_pre,
                                                               n=self.poly, model=None)
        else:
            self.X_poly = self.X_pre
        
        if self.stsc:
            self.X_sc, self.models['stsc'] = standard_scaler(self.X_poly)
        else:
            self.X_sc = self.X_poly


if __name__ == '__main__':
    print("hello")