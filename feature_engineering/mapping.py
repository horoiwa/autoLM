
def simple_mapping(dataframe, criterio=15):
    fmap = {"const": [],
            "numeric": [],
            "ordinal": [],
            "category": []}
    
    for col in dataframe.columns:
        data = dataframe[col]

        if len(set(data)) == 1:
            fmap["const"].append(col)
        
        elif len(set(data)) < criterio:
            try:
                [float(val) for val in data]
                fmap["ordinal"].append(col)
            except ValueError:
                fmap["category"].append(col)
        else:
            try:
                [float(val) for val in data]
                fmap["numeric"].append(col)
            except ValueError:
                print("Invalid feature: {}".format(col))
                raise ValueError

    return fmap
