from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split


class RidgeRPRS():
    """Ridge with Random Patches and Random Subspaces
    """
    def __init__(self, dataset):
        self.dataset = dataset

        self.X = self.dataset.X_sc
        self.y = self.dataset.y
        
        self.colummns = {}
        self.indices = {}
        self.models = {}

    def run_ridge(self, n=100):
        self.ridge_rprs(self.X, self.y, test_set=None)

    def run_ridge_evaulate(self, n=100):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y)
        self.ridge_rprs(X_train, y_train, test_set=(X_test, y_test))

    def ridge_rprs(self, X, y, test_set=None):
        if not test_set:
            print("record model")

        else:
            print("evaluate model")
        