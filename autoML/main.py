import shutil
import os
import pickle

import numpy as np

from autoML.support import load_df, load_sample
from autoML.dataset import DataSet
from autoML.feature_selection import FeatureSelectionGA
from autoML.extended_linear_model import RidgeRPRS

def main():
    """GAtest
    """
    project_name = "sample project"
    if os.path.exists(project_name):
        shutil.rmtree(project_name)
    
    X, y = load_df("boston")
    X_sample, y_sample = load_sample('boston') 

    dataset = DataSet(project_name, poly=1)
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


def main2():
    """
        test for feature screening
    """
    pass


def main3():
    """
        test for extended linear models
    """
    project_name = "sample project"
    if os.path.exists(project_name):
        shutil.rmtree(project_name)
    
    X, y = load_df("boston")
    X_sample, y_sample = load_sample('boston') 

    dataset = DataSet(project_name, poly=1)
    print(dataset)
    dataset.fit(X, y)

    model = RidgeRPRS(dataset, n_models=1000)
    model.evaluate()

    model.fit()
    mean, std = model.predict(dataset.transform(X_sample))
    print("Pred:", mean, std)
    print("Obs:", y_sample)

if __name__ == '__main__':
    #main()
    #main2()
    main3()