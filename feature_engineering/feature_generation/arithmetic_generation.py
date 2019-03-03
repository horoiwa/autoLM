import copy
import itertools
import numpy as np
import pandas as pd

from feature_engineering.util import feature_map


def square_generation(df, fmap):
    df_temp = copy.deepcopy(df[fmap["numerical"]])
    df_square = df_temp**2
    df_square.columns = [col+"_sq" for col in df_square.columns]
    return df_square


def log_generation(df, fmap):
    df_temp = copy.deepcopy(df[fmap["numerical"]])
    df_log = np.log(df_temp)
    df_log.columns = [col+"_log" for col in df_log.columns]
    return df_log


def cross_generation(df, fmap):
    df_cross = pd.DataFrame()
    for col_x, col_y in itertools.combinations(df.columns, r=2):
        col_name = col_x + "_" + col_y
        df_cross[col_name] = df[col_x]*df[col_y] 

    return df_cross


if __name__ == '__main__':
    from feature_engineering.util.sample_data import _generate_testdf 
    df = _generate_testdf(rows=50)
    fmap = feature_map.assign_featuretype(df)

    df_square = square_generation(df, fmap)
    df_log = log_generation(df, fmap)
    df_cross = cross_generation(df, fmap)

    print(df_square.shape, df_log.shape, df_cross.shape)
