import pandas as pd
from sklearn.preprocessing import (PolynomialFeatures, OneHotEncoder, 
                                   StandardScaler)


def onehot_conversion(df, model=None):
    if model:
        df_onehot = pd.DataFrame(model.transform(df),
                                 columns=model.get_feature_names(
                                    input_features=df.columns))
        df_onehot.index = df.index
        return df_onehot

    model = OneHotEncoder(sparse=False)
    model.fit(df)
    df_onehot = pd.DataFrame(model.transform(df),
                             columns=model.get_feature_names(
                                input_features=df.columns)
                            )
    df_onehot.index = df.index

    cmap = {}
    for col in df.columns:
        cmap[col] = list(pd.get_dummies(df[col]).columns)

    return df_onehot, model, cmap


def poly_generation(dataframe, n=3, model=None):
    if model:
        df_poly = pd.DataFrame(model.transform(dataframe),
                               columns=model.get_feature_names(
                                   input_features=dataframe.columns))
        return df_poly

    model = PolynomialFeatures(degree=n)
    model.fit(dataframe)
    df_poly = pd.DataFrame(model.transform(dataframe),
                           columns=model.get_feature_names(
                                input_features=dataframe.columns)) 
    return df_poly, model


def standard_scaler(dataframe, model=None):
    if model:
        X_stsc = model.transform(dataframe)
        df_stsc = pd.DataFrame(X_stsc, columns=dataframe.columns, index=dataframe.index)
        return df_stsc

    model = StandardScaler()
    model.fit(dataframe)
    X_stsc = model.transform(dataframe)
    df_stsc = pd.DataFrame(X_stsc, columns=dataframe.columns, index=dataframe.index)
    
    return df_stsc, model
