from sklearn.datasets import load_iris, load_boston
import pandas as pd


def load_dataset(dataset="boston"):
    """ Load sample dataset from sklearn as pandas DataFrame
    """
    if dataset == "boston":
        data = load_boston()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        return df
    elif dataset == "iris":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        return df
    else:
        print("Error: Unexpected dataset name")
        raise NotImplementedError


if __name__ == '__main__':
    df = load_dataset(dataset="boston")
    print(df.head())

    df = load_dataset(dataset="iris")
    print(df.head())