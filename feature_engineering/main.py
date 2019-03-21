from feature_engineering.support import load_df, load_sample
from feature_engineering.dataset import DataSet


if __name__ == '__main__':
    X, y = load_df("boston")
    sample_x, sample_y = load_sample('boston') 

    dataset = DataSet(X, y)
    
    print()
    print("Dataset check")
    print(dataset.X.head(2))
    print(dataset.X.shape)
    dataset._preprocess()

    print()
    print("Structurization")
    print(dataset.X_pre.head(2))
    print(dataset.X_pre.shape)

    print()
    print("Generation")
    dataset._postporcess()
    print(dataset.X_post.head(2))
    print(dataset.X_post.shape)
    

    #print(sample_x)