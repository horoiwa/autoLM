import os
import copy
import pickle

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


class RidgeRPRS():
    """Ridge with Random Patches and Random Subspaces
    """
    def __init__(self, dataset, n_models):
        self.dataset = dataset
        self.project_name = dataset.project_name

        self.X = self.dataset.get_X_processed()
        self.y = self.dataset.y
        self.all_columns = self.X.columns
        
        self.n_models = n_models
        
        self.use_columns = {}
        self.models = {}
    
    def print_data(self):
        pass

    def fit(self, columns_ratio=0.4, samples_ratio=0.4):
        self.ridge_rprs(self.X, self.y,
                        columns_ratio=columns_ratio,
                        samples_ratio=samples_ratio
                        )
        
        savepath = os.path.join(self.project_name, 'RidgeRPRS.pkl')
        with open(savepath, 'wb') as f:
            pickle.dump(self, f)

    def evaluate(self, columns_ratio=0.4,
                 samples_ratio=0.4, test_size=0.3): 
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                            test_size=test_size)
        self.ridge_rprs(X_train, y_train,
                        columns_ratio=columns_ratio,
                        samples_ratio=samples_ratio
                        )

        pred_mean, pred_std = self.predict(X_test)
        rscore = st.pearsonr(pred_mean.values.reshape(-1, 1), y_test)
        r2score = rscore[0][0]**2

        print("Evaluated r2 score", r2score)

        self.use_colummns = {}
        self.models = {}

    def predict(self, X):
        """
            X must be DataFrame with columns
        """
        assert np.all(X.columns == self.X.columns), "Insconsitent columns"
        assert len(self.models.values()) > 0, "No trained models"

        df_result = pd.DataFrame() 
        for i in range(self.n_models):
            model = self.models[i]
            cols = self.use_columns[i]
            y_pred = model.predict(X[cols])
            df_result[i] = list(y_pred)
        
        mean = df_result.mean(1)
        std = df_result.std(1)

        if len(mean) == 1:
            mean = mean[0]
            std = std[0]

        return mean, std

    def ridge_rprs(self, X, y,
                   columns_ratio, samples_ratio):
        for i in range(self.n_models):
            X_rp, X_, y_rp, y_ = train_test_split(X, y, train_size=samples_ratio)
            mask_cols = [bool(np.random.binomial(1, columns_ratio)) for col in X.columns]
            use_cols = X.columns[mask_cols]

            X_rprs = X_rp[use_cols]
            self.use_columns[i] = copy.deepcopy(use_cols)
            model = RidgeCV()
            model.fit(X_rprs, y_rp)
            self.models[i] = copy.deepcopy(model)