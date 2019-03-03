from sklearn.datasets import load_iris, load_boston
import pandas as pd


def load_df(dataset=None):
    """ Load sample dataset from sklearn as pandas DataFrame
    """
    if dataset == "boston":
        df = pd.DataFrame(load_boston().data,
                          columns=load_boston().feature_names)
        return df
    elif dataset == "iris":
        df = pd.DataFrame(load_iris().data,
                          columns=load_iris().feature_names)
        return df
    else:
        print("Error: Unexpected dataset name")
        raise NotImplementedError


if __name__ == '__main__':
    df = load_df(dataset="boston")
    print(df.head())

    df = load_df(dataset="iris")
    print(df.head())