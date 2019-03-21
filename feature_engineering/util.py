import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


def onehot_conversion(dataframe):
    dataframe_onehot = pd.get_dummies(dataframe)

    cmap = {}
    for col in dataframe.columns:
        cmap[col] = list(pd.get_dummies(dataframe[col]).columns)

    return dataframe_onehot, cmap


def poly_generation(dataframe, n=2):
    model_poly = PolynomialFeatures(degree=n)
    model_poly.fit(dataframe)
    dataframe_poly = pd.DataFrame(model_poly.transform(dataframe),
                                  columns=model_poly.get_feature_names(
                                      input_features=dataframe.columns)) 
    return dataframe_poly, model_poly
