import os

import pickle
import pandas as pd
import numpy as np

from autoML.util import (onehot_conversion, poly_generation,
                         standard_scaler, simple_mapping)


class DataSet():
    def __init__(self, project_name, criterio=15, poly=1, stsc=True):
        self.project_name = project_name
        self._create_project_dir()        
        
        self.X = None
        self.y = None
        self.models = {}

        self.fmap = None
        self.cmap = None

        # データ途中経過
        self.X_pre = None
        self.X_poly = None
        self.X_sc = None

        # pipeline config
        self.criterio = criterio
        self.poly = poly
        self.stsc = stsc

        # final counter
        self.fit_count = 0

    def __repr__(self):
        return "DataSet Object"

    def input_check(self):
        assert isinstance(self.X, 
                          (pd.DataFrame, pd.Series)
                          ), "input must be DataFrame or Series"
        assert isinstance(self.y, 
                          (pd.DataFrame, pd.Series)
                          ), "input must be DataFrame or Series"

        assert self.X.shape[0] == self.y.shape[0], "Error #1"
        assert len(self.X.columns) > 0, "Error #2"
        assert len(self.y.columns) == 1, "Y should have just one column"

    def transform(self, X):
        """ 実装予定
            引き算割り算記述子の追加
        """
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
        assert self.fit_count == 0, "Fit can use only once"
        
        self.X = X
        self.y = y
        self.input_check()

        self._preprocess()
        self._postprocess()

        self._save_dataset()
        self.fit_count += 1

    def _save_dataset(self):
        savepath = os.path.join(self.project_name, 'dataset.pkl')
        with open(savepath, 'wb') as f:
            pickle.dump(self, f)

    def get_X_processed(self):
        """Return converted dataset
        """
        if np.any(self.X_sc):
            return self.X_sc
        else:
            print("Dataset is Empty: use fit method")

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

    def _create_project_dir(self):
        def rename_project(name, n):
            new_name = name + "_({})".format(n)  
            if os.path.exists(new_name):
                new_name = rename_project(name, n+1)
            return new_name

        if os.path.exists(self.project_name):
            self.project_name = rename_project(self.project_name, 1)
            print("Warning: project_name is dupilicated and renamed to {}".format(self.project_name))
        os.makedirs(self.project_name)
        

if __name__ == '__main__':
    print("hello")