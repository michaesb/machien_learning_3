from sklearn.preprocessing.data import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def retrieve_data( undersampling=False, ratio = 1):
    path = os.path.dirname(os.path.realpath(__file__))
    file1 = path + "/../data/creditcard_part1.csv"
    file2 = path + "/../data/creditcard_part2.csv"

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df  = pd.concat((df1, df2), ignore_index=True)


    class_counts = df.Class.value_counts()
    num_fraudulent = class_counts[1]
    num_non_fraudulent = class_counts[0]


    ## There are no categories in the dataset, so no need to do one-hot encoding.
    X = df.loc[:, df.columns != 'Class'].values
    y = df.loc[:, df.columns == 'Class'].values.ravel()


    #### StandardScaler is more useful for classification, and Normalizer is more useful for regression.
    standard_scaler = StandardScaler()
    X = standard_scaler.fit_transform(X)


    ### Do undersampling to fix imbalanced class
    if undersampling:
        np.random.seed(1)
        if ratio > 1:
            raise ValueError("ratio cannot be bigger than one")

        indices_nonfraud = np.where(y==0)[0]
        indices_fraud = np.where(y==1)[0]

        multiplier = int(1.0/ratio)

        np.random.shuffle(indices_nonfraud)
        indices_nonfraud_under = indices_nonfraud[:multiplier*num_fraudulent]
        indices_under = np.concatenate( (indices_fraud, indices_nonfraud_under) )
        np.random.shuffle(indices_under)

        X = X[indices_under]
        y = y[indices_under]
    ####

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4)
    return X_train,X_test,y_train,y_test

if __name__ == '__main__':

    X_train,X_test,y_train,y_test = retrieve_data(undersampling=True, ratio=0.5)
    print (len(y_train))
    print (len(y_test))
    print ()
