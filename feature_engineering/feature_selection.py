from feature_engineering.dataset import DataSet


class FeatureSelectionGA():
    def __init__(self, DataSet, n_features=(None, None)):
        self.DataSet = DataSet

        self.init_check()

    def init_check(self):
        assert self.DataSet.__repr__() == DataSet().__repr__(), "Error #21"

    def set_selected_feature(self):
        self.DataSet.selected_features = "Selected"
        self.DataSet.X_fin = self.DataSet.X_sc["Selected"]