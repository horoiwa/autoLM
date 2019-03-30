from sklearn.linear_model import Ridge


class RidgeRPRS():
    """Ridge with Random Patches and Random Subspaces
    """
    def __init__(self, dataset):
        self.dataset = dataset
        