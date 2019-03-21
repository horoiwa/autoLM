import pandas as pd

from feature_engineering.mapping import simple_mapping 


class DataSet():
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.fmap = None

        self.X_2 = None
        self.fmap_2 = None

    def preprocess(self, criterio=15):
        self.fmap = simple_mapping(self.X, criterio=criterio)

        X_num = self.X[self.fmap["numeric"]]
        X_ord = self.X[self.fmap["ordinal"]]
        X_cat = self.X[self.fmap["category"]]

        X_onehot = pd.get_dummies(X_cat) 




if __name__ == '__main__':
    print("hello")