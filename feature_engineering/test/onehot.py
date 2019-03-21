import copy
import pandas as pd
import numpy as numpy


def one_hot_conversion(df, fmap):
    """ one hot encoding of categorical features 
        Parameters
        ----------
        df : DataFrame 

        fmap :  Dict
            faturetype map

        Return
        ----------
        df : DataFrame
            one-hot encoded data

        category_map : Dict
            カテゴリ変数の排他性を記録
            最適化までするなら役に立つ
    """
    df_cat = df[fmap['category']]
    df = copy.deepcopy(df.drop(fmap['category'], 1))

    category_map = {}

    for col in df_cat.columns:
        df_temp = pd.get_dummies(df_cat[col])
        df = pd.concat([df, df_temp], 1)
        category_map[col] = df_temp.columns

    return df, category_map




if __name__ == '__main__':
    from .sample_data import _generate_testdf
    from .feature_map import assign_featuretype
    df = _generate_testdf(50)
    fmap = assign_featuretype(df)
    df, category_map = one_hot_conversion(df, fmap)

    print(df.columns)
    print(category_map)
