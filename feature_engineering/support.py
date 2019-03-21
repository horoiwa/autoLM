from sklearn.datasets import load_iris, load_boston
import pandas as pd
import numpy as np
import random


def load_df(dataset=None):
    """ Load sample dataset from sklearn as pandas DataFrame
    
    Return:
    X : input features && dataframe
    y : objective && dataframe
    """
    if dataset == "boston":
        X = pd.DataFrame(load_boston().data,
                          columns=load_boston().feature_names)
        
        dummy = [random.choice(["Austin", "Houston", "Dallas"]) for _ in range(X.shape[0])]
        X_dummy = pd.DataFrame(np.array(dummy).reshape(-1,1), columns=["USA"])
        X = pd.concat([X, X_dummy], 1)

        dummy = [random.choice(["Tokyo", "Kyoto", "Sapporo"]) for _ in range(X.shape[0])]
        X_dummy = pd.DataFrame(np.array(dummy).reshape(-1,1), columns=["Japan"])
        X = pd.concat([X, X_dummy], 1)
        
        dummy = [random.choice(["Armadiilo"]) for _ in range(X.shape[0])]
        X_dummy = pd.DataFrame(np.array(dummy).reshape(-1,1), columns=["Animal"])
        X = pd.concat([X, X_dummy], 1)
        
        y = pd.DataFrame(load_boston().target.reshape(-1, 1), columns=['Price'])
        return X, y

    elif dataset == "iris":
        X = pd.DataFrame(load_iris().data,
                          columns=load_iris().feature_names)
        y = pd.DataFrame(load_iris.traget.reshape(-1, 1), columns=["Iris"])
        return X, y
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
    categorical2 = [random.choice(["E", "F", "G", "H"]) for _ in range(rows)]
    binary = [random.choice([0, 1]) for _ in range(rows)]
    
    df["num"] = numerical
    df["num2"] = numerical
    df["cat"] = categorical
    df["cat2"] = categorical2
    df["bin"] = binary
    df["bin2"] = binary
    df["ord"] = ordinal
    df["ord2"] = ordinal

    y = "dummy"
    return df, y
    

if __name__ == '__main__':
    df = load_df(dataset="boston")
    print(df.head())

    df = load_df(dataset="iris")
    print(df.head())

    df = _generate_testdf(rows=50)
    print(df.head())