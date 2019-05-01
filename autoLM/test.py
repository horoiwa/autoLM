import shutil
import os
import pickle

import numpy as np

from autoLM.support import load_df, load_sample
from autoLM.dataset import DataSet
from autoLM.feature_selection import FeatureSelectionGA
from autoLM.extended_linear_model import RidgeRPRS
from autoLM.feature_screening import FeatureScreeningGA


def test1():
    """GAtest
    """
    project_name = "sample project"
    if os.path.exists(project_name):
        shutil.rmtree(project_name)

    X, y = load_df("boston")
    X_sample, y_sample = load_sample('boston')

    dataset = DataSet(project_name, poly=2)
    print(dataset)
    dataset.fit(X, y)
    X_test1 = dataset.get_X_processed()
    X_test2 = dataset.transform(X)
    X_test1.to_csv("test.csv")

    assert np.all(X_test1.values == X_test2.values)
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
    print(X_post.head())
    X_post_sample.to_csv("sample.csv")

    print()
    print("Run Feature selection By GA")
    print()
    ga_selecter = FeatureSelectionGA(DataSet=dataset, n_features=(10, 20))
    ga_selecter.run_RidgeGA(n_gen=3, n_eval=10)
    print("GA finished")

    with open('../dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)

    with open('../dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)

    print(ga_selecter.ga_result)
    for key in ga_selecter.selected_features.keys():
        print()
        print(ga_selecter.selected_features[key].columns)
        print()

    return dataset


def test2(dataset):
    """
        test for feature screening
    """
    screening = FeatureScreeningGA(dataset, n_features=(5, 10),
                                   n_gen=3, n_eval=10)
    screening.run(prescreening=4, postscreening=4, n_jobs=4)


def test3(dataset):
    """
        test for extended linear models
    """
    X_sample, y_sample = load_sample('boston')

    model = RidgeRPRS(dataset, n_models=100)
    model.evaluate()

    model.fit()
    mean, std = model.predict(dataset.transform(X_sample))
    print("Pred:", mean, std)
    print("Obs:", y_sample)


if __name__ == '__main__':
    test1()
    test3()
    test2()
