import copy
import os
import collections

from joblib import Parallel, delayed

from autoLM.feature_selection import FeatureSelectionGA


class FeatureScreeningGA():

    def __init__(self, dataset, n_features=(None, None), n_gen=3, n_eval=10):
        self.dataset = dataset
        self.n_features = n_features

        self.n_gen =3
        self.n_eval = 10

        self.logdir = os.path.join(self.dataset.project_name, "FeatureScreening")

    def run(self, prescreening=20, postscreening=6, n_jobs=1):
        os.makedirs(self.logdir)
        prescreen_results = Parallel(n_jobs=n_jobs)([delayed(self._ga_selection)("1st_No{}".format(i)) for i in range(prescreening)])
        self._print_log("1st_screening.log", prescreen_results)
        self.usecols = self._get_unique(prescreen_results)
        print(self.usecols)

        postscreen_results = Parallel(n_jobs=n_jobs)([delayed(self._ga_selection)("2nd_No{}".format(i), self.usecols) for i in range(postscreening)])
        self._print_log("2nd_screening.log", postscreen_results)
        print(self._get_unique(postscreen_results))

        self._create_summary(postscreen_results)
    
    def _create_summary(self, postscreen_results):
        selected_features = []
        for _list in postscreen_results:
            selected_features.extend(_list)

        summary = collections.Counter(selected_features)
        with open(os.path.join(self.logdir, 'summary.txt'), 'a') as f:
            for name, count in summary.items():
                f.write("{},{},\n".format(name, count))

    def _print_log(self, fname, results):
        with open(os.path.join(self.logdir, fname), 'a') as f:
            unique_features = self._get_unique(results) 
            f.write("---"*12+"\n")
            f.write(",".join(unique_features))
            f.write("\n")
            f.write("---"*12+"\n")
            for result in results:
                line = ",".join(result)
                f.write(line+"\n")
    
    def _get_unique(self, lists_in_list):
        features = []
        for _list in lists_in_list:
            for feature in _list:
                features.append(feature)

        unique = list(set(features))
        return unique
         
    def _ga_selection(self, name, usecols=None):
        ga_selecter = FeatureSelectionGA(DataSet=dataset, 
                                         n_features=self.n_features)
        if usecols:
            ga_selecter.set_usecols(usecols)

        ga_selecter.run_RidgeGA(n_gen=self.n_gen, n_eval=self.n_eval)

        with open(os.path.join(self.logdir, name+".txt"), 'a') as f:

            f.write("GA_"+name+"\n")
            f.write("Run Feature selection By GA \n")
            f.write("\n")
            df_result = ga_selecter.ga_result
            keys = ga_selecter.selected_features.keys()
            print(df_result)
            for n, key in enumerate(keys):
                f.write("\n")
                f.write("N_features: " + str(key) + " \n")
                f.write("Score {}".format(str(df_result.values[n, 1])))
                f.write("\n")
                f.write("\n")
                features = list(ga_selecter.selected_features[key].columns)
                f.write("["+",".join(features)+"]")
                f.write("\n")
                f.write("\n")

            f.write("GA finished \n")
            
            key_min = min([int(key) for key in keys])
            min_features = list(ga_selecter.selected_features[key_min].columns)
        return copy.deepcopy(min_features)


if __name__ == '__main__':
    import shutil

    from autoLM.support import load_df
    from autoLM.dataset import DataSet

    project_name = "sample project"
    if os.path.exists(project_name):
        shutil.rmtree(project_name)

    X, y = load_df("boston")

    dataset = DataSet(project_name, poly=2)
    dataset.fit(X, y)
    screening = FeatureScreeningGA(dataset, n_features=(5, 10),
                                   n_gen=10, n_eval=50)
    screening.run(prescreening=30, postscreening=9, n_jobs=3)