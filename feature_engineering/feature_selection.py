from feature_engineering.dataset import DataSet


class FeatureSelectionGA():
    def __init__(self, DataSet=None, X=None, y=None, n_features=(None, None)):

        self.DataSet = DataSet

        if self.DataSet:
            self.X = self.DataSet.X_sc
            self.y = self.DataSet.y
        else:
            self.X = X
            self.y = y

        self.min_features = n_features[0]
        self.max_features = n_features[1]

        self.initial_check()

    def initial_check(self):
        if self.DataSet:
            assert self.DataSet.__repr__() == DataSet().__repr__(), "Error #21"
        assert self.min_features, "n_features required"       
        assert self.max_features, "n_features required"
        assert self.max_features > self.min_features, "max min"

    def set_selected_feature(self):
        self.DataSet.selected_features = "Selected"
        self.DataSet.X_fin = self.DataSet.X_sc["Selected"]