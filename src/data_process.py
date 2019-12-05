from sklearn.preprocessing.data import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
def retrieve_data( undersampling=False ):
    path = os.path.dirname(os.path.realpath(__file__))
    file1 = path + "/../data/creditcard_part1.csv"
    file2 = path + "/../data/creditcard_part2.csv"

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df  = pd.concat((df1, df2), ignore_index=True)


    class_counts = df.Class.value_counts()
    num_fraudulent = class_counts[1]
    num_non_fraudulent = class_counts[0]


    ### Undersampling
    if undersampling:
        indices = np.arange(0, num_fraudulent)

        frauds_df = df.loc[ df["Class"] == 1 ]
        non_frauds_df = df.loc[ df["Class"] == 0 ]

        indices = np.arange(0, num_non_fraudulent)
        indices = np.random.choice(indices, num_fraudulent, replace=False)

        #non_frauds_df = non_frauds_df.loc[ indices ]
        non_frauds_df = non_frauds_df.reindex( indices )

        under_df  = pd.concat((frauds_df, non_frauds_df), ignore_index=True)
        df = under_df.sample(frac=1).reset_index(drop=True)
    ####


    ## There are no categories in the dataset, so no need to do one-hot encoding.
    X = df.loc[:, df.columns != 'Class'].values
    y = df.loc[:, df.columns == 'Class'].values.ravel()


    #### StandardScaler is more useful for classification, and Normalizer is more useful for regression.
    standard_scaler = StandardScaler()
    X = standard_scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4)
    return X_train,X_test,y_train,y_test

    ### Do undersampling to fix imbalanced class
if __name__ == '__main__':

    X,y = retrieve_data()
    print ("ohh, fuck" )
    print (X)
    print ("I can't believe you've done this")
    print (np.sum(y)/len(y)*100)
