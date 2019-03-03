from sklearn.datasets import load_iris, load_boston
import pandas as pd
import numpy as np
import random


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


def _generate_testdf(rows=50):
    """ Generate test dataset
    """
    df = pd.DataFrame()

    numerical = [np.random.uniform(100) for _ in range(rows)]
    ordinal = [random.choice([1, 2, 3, 4, 5]) for _ in range(rows)]
    categorical = [random.choice(["A", "B", "C", "D"]) for _ in range(rows)]
    binary = [random.choice([0, 1]) for _ in range(rows)]
    
    df["num"] = numerical
    df["cat"] = categorical
    df["bin"] = binary
    df["ord"] = ordinal

    return df
    

if __name__ == '__main__':
    df = load_df(dataset="boston")
    print(df.head())

    df = load_df(dataset="iris")
    print(df.head())

    df = _generate_testdf(rows=50)
    print(df.head())