import random

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from feature_engineering.dataset import DataSet


class FeatureSelectionGA():
    def __init__(self, DataSet=None, X=None, y=None, n_features=(None, None)):

        self.DataSet = DataSet

        if self.DataSet:
            self.X = self.DataSet.X_sc
            self.y = self.DataSet.y
        else:
            self.X = X
            self.y = y

        self.min_features = n_features[0]
        self.max_features = n_features[1]

        self.initial_check()

    def initial_check(self):
        if self.DataSet:
            assert self.DataSet.__repr__() == DataSet().__repr__(), "Error #21"
        assert self.min_features, "n_features required"       
        assert self.max_features, "n_features required"
        assert self.max_features > self.min_features, "max min"

    def run_RidgeGA(self, n_gen, n_eval):
        self.ridgeGA = RidgeGA(X=self.X, y=self.y, max_features=self.max_features,
                               min_features=self.min_features, n_gen=n_gen, n_eval=n_eval)
        
        df_result = self.ridgeGA.run()
        self.set_selected_features(df_result)
        
    def set_selected_features(self, df_result):
        df_n_score = df_result[["N", "SCORE"]]
        df_features = df_result.drop(["N", "SCORE"], 1)
        
        self.DataSet.ga_result = df_n_score
        
        for idx in df_n_score.index:
            n = df_n_score.loc[idx, 'N']

            temp = df_features.loc[idx, :]
            selected_features = df_features.columns[[bool(val) for val in temp]]

            self.DataSet._selected_features[n] = self.X[selected_features]


class RidgeGA():
    """ GAによる特徴量選択
        まずは単純にonemax問題を
    """
    def __init__(self, X, y, max_features, min_features, n_gen, n_eval):
        self.X = X
        self.y = y

        self.weights = (-1.0, 1.0)
        self.n_eval = n_eval
        self.n_gen = n_gen

        self.pop = None
        self.log = None
        self.hof = None
        self.result = None

        self._MAX_FEATURES = max_features
        self._MIN_FEATURES = min_features
        self._TOTAL_FEATURES = self.X.shape[1]

    def eval_score(self, X, n):
        """ RidgeCV
            Parameters
            -------------
            X: pandas dataframe
            n: train_test_splitの回数

            Return
            -------------
            score: average score
        """
        scores = []
        for _ in range(n):
            X_train, X_test, y_train, y_test = train_test_split(X, self.y, test_size=0.4) 
            model = RidgeCV()
            model.fit(X_train, y_train)
            scores.append(model.score(X_test, y_test))

        score = np.array(scores).mean()
        return score

    def run(self):
        """ Feature optimization by NSGA-2
            max_item means max_feature

        """
        def generateIndividual(container, n):
            ratio = random.uniform(self._MIN_FEATURES/n,
                                   self._MAX_FEATURES/n)
            generated_val = [np.random.binomial(1, ratio) for _ in range(n)]
            individual = container(generated_val)
            return individual

        def evalIndividual(individual):
            n_features = sum(individual)
            
            if n_features == 0:
                return 9999, -9999
            elif n_features > self._MAX_FEATURES:
                return 9999, -9999
            elif n_features < self._MIN_FEATURES:
                return 9999, -9999
            else:
                X_temp = self.X.iloc[:, [bool(val) for val in individual]]
                score = self.eval_score(X_temp, self.n_eval)

            # print(n_features, " ", score)
            return n_features, score

        def main():
            NGEN = self.n_gen
            MU = 200
            LAMBDA = 500
            CXPB = 0.7
            MUTPB = 0.1

            pop = toolbox.population(n=MU)
            hof = tools.ParetoFront()
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("min", np.min, axis=0)
            stats.register("max", np.max, axis=0)
           
            pop, log = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB,
                                                 MUTPB, NGEN, stats, halloffame=hof)

            return pop, log, hof


        # 特徴数を最小化　精度を最大化
        creator.create("Fitness", base.Fitness, weights=self.weights)   
        creator.create("Individual", list, fitness=creator.Fitness)

        toolbox = base.Toolbox()

        toolbox.register("individual", generateIndividual, creator.Individual, self.X.shape[1])
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", evalIndividual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selNSGA2)

        self.pop, self.log, self.hof = main()
        self.result = self.create_result()
        print("GA finised gracefully")
        
        return self.result

    def create_result(self):
        scores = []
        n_features = []
        for ind in self.hof:
            n_features.append(sum(ind))
            X_temp = self.X.iloc[:, [bool(val) for val in ind]]
            score = self.eval_score(X_temp, 200)
            scores.append(score)

        X = pd.DataFrame(np.array(self.hof), columns=self.X.columns) 
        scores = pd.DataFrame(np.array(scores), columns=["SCORE"])
        n_features = pd.DataFrame(np.array(n_features), columns=["N"])
        try:
            result = pd.concat([n_features, scores, X], 1)
        except TypeError:
            X.to_csv("../error_X.csv")
            score.to_csv("../error_score.csv")
            n_features.to_csv("../error_n.csv")

        return result
