from feature_engineering.support import load_df, load_sample
from feature_engineering.dataset import DataSet
from feature_engineering.feature_selection import FeatureSelectionGA
import pickle
import numpy as np


if __name__ == '__main__':
    X, y = load_df("boston")
    X_sample, y_sample = load_sample('boston') 

    dataset = DataSet(poly=3)
    print(dataset)
    dataset.fit(X, y)

    print()
    print("PICKLE TEST")
    print()
    with open('../dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)

    with open('../dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)

    X_post = dataset.transform(X)
    X_post_sample = dataset.transform(X_sample)

    print("Consistency check")
    print(np.all(X_post.columns == X_post_sample.columns))
    print(X_post.head())
    print(X_post_sample)

    print()
    print("Run Feature selection By GA")
    print()
    ga_selecter = FeatureSelectionGA(DataSet=dataset, n_features=(30, 50))
    ga_selecter.run_RidgeGA(n_gen=10, n_eval=3)
    print("GA finished")


