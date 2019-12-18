#!/usr/bin/env python

## Importing needed packages
from sklearn.preprocessing.data import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

def retrieve_data( undersampling=False, ratio = 1, random_state=None):
    ## Getting and reading csv-data files into a pandas dataframe
    path = os.path.dirname(os.path.realpath(__file__))
    file1 = path + "/../data/creditcard_part1.csv"
    file2 = path + "/../data/creditcard_part2.csv"

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df  = pd.concat((df1, df2), ignore_index=True)

    ## Finding the class balances
    class_counts = df.Class.value_counts()
    num_fraudulent = class_counts[1]
    num_non_fraudulent = class_counts[0]

    ## Splitting the dataset into design matrix X and targets y
    X = df.loc[:, df.columns != 'Class'].values
    y = df.loc[:, df.columns == 'Class'].values.ravel()

    #### StandardScaler is more useful for classification, and Normalizer is more useful for regression.
    standard_scaler = StandardScaler()
    X = standard_scaler.fit_transform(X)

    ### Undersampling to fix imbalanced class
    if undersampling:

        if random_state is not None:
            np.random.seed(random_state)
            
        if ratio > 1:
            raise ValueError("Undersampling ratio can't be larger than one")
        
        multiplier = int(1.0/ratio)

        ## Randomized undersampling method
        indices_nonfraud = np.where(y==0)[0]
        indices_fraud = np.where(y==1)[0]
        np.random.shuffle(indices_nonfraud)
        indices_nonfraud_under = indices_nonfraud[:multiplier*num_fraudulent]
        indices_under = np.concatenate( (indices_fraud, indices_nonfraud_under) )
        np.random.shuffle(indices_under)

        ## Using indices from undersampling method to create new balanced dataset
        X_under = X[indices_under]
        y_under = y[indices_under]

    ## Splitting the dataset into test and training sets
    X_train, X_test, y_train, y_test = train_test_split(X_under, y_under, test_size=0.33, random_state=4)
    return X_train,X_test,y_train,y_test


## For testing purposes
if __name__ == '__main__':
    X_train, X_test, y_train, y_test = retrieve_data(undersampling=True, ratio=1.0)
    print (len(y_train))
    print (len(y_test))
