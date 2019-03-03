import pandas as pd
import numpy as numpy


def one_hot_conversion(df, fmap):

    return df




if __name__ == '__main__':
    from .sample_data import _generate_testdf
    from .feature_map import assign_featuretype
    df = _generate_testdf(50)
    fmap = assign_featuretype(df) 
    print(fmap)