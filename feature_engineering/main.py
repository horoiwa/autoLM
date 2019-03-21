from feature_engineering.support import load_df
from feature_engineering.dataset import DataSet


if __name__ == '__main__':
    X, y = load_df("boston")

    dataset = DataSet(X, y)
    print("Dataset check")
    print(dataset.X.head())
    print()
    dataset.preprocess()
    print(dataset.fmap)