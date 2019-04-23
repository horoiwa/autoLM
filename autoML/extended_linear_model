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

    def run_ridge(self, n_features=None, n_models=100):
        self.ridge_rprs(train_set=(self.X, self.y),
                        test_set=None,
                        n_features=n_features,
                        n_models=n_models)

    def run_ridge_evaulate(self, n_features=None, n_models=100):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y)
        self.ridge_rprs(train_set=(X_train, y_train),
                        test_set=(X_test, y_test),
                        n_features=n_features,
                        n_models=n_models)

    def ridge_rprs(self, train_set, test_set, n_features, n_models):
        X = train_set[0]
        y = train_set[1]

        print(X.columns)
        print(y.shape)

        for _ in range(n_models):
            print(_)

        if not test_set:
            print("record model")

        else:
            print("evaluate model")
        