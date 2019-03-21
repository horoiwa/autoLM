import pandas as pd

from feature_engineering.mapping import simple_mapping
from feature_engineering.util import onehot_conversion, poly_generation


class DataSet():
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.pipeline = {}

        self.fmap = None
        self.cmap = None

        self.X_pre = None
        self.X_post = None

        self.initial_check()

    def initial_check(self):
        assert self.X.shape[0] == self.y.shape[0], "Error #1"
        assert len(self.X.columns) > 0, "Error #2"
        assert len(self.y.columns) > 0, "Error #3"

    def fit_sample(self, X_sample):
        assert X_sample.columns == self.X.columns

    def fit(self):
        self._preprocess()
        self._postporcess()

    def _preprocess(self, criterio=15):
        self.fmap = simple_mapping(self.X, criterio=criterio)

        X_num = self.X[self.fmap["numeric"]]
        X_ord = self.X[self.fmap["ordinal"]]
        X_cat = self.X[self.fmap["category"]]

        X_onehot, self.cmap = onehot_conversion(X_cat)

        self.X_pre = pd.concat([X_num, X_ord, X_onehot], 1)

    def _postporcess(self):
        self.X_post, self.pipeline["poly"] = poly_generation(self.X_pre)



if __name__ == '__main__':
    print("hello")