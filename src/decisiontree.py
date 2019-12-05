#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
import time
from sklearn import tree
from sklearn.metrics import accuracy_score, recall_score, precision_score

from data_process import retrieve_data

"""
runs the classifier class with data from the data package
"""

def neuralnet_clf_sklearn():
    ratio_ = 0.25
    X_train, X_test, y_train, y_test = retrieve_data(undersampling = True, ratio = ratio_)


    learning_rate = np.linspace(0.01, 0.001, 11)
    n = len(learning_rate)
    timer = np.zeros(n)
    acc_score = np.zeros(n)
    rec_score = np.zeros(n)
    prec_score = np.zeros(n)
    for i in range(len(learning_rate)):
        time1 = time.time()
        print(int(100*i/len(learning_rate)), "%")

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train.ravel())
        predict = clf.predict(X_test)
        acc_score[i] = accuracy_score(y_test.ravel(),predict)
        prec_score[i] = precision_score(y_test.ravel(),predict)
        rec_score[i] = recall_score(y_test.ravel(),predict)
        time2 = time.time()
        timer[i] =time2 -time1
        # print("time = ",timer[i]," s")
    print ("ratio: ",ratio_)
    print("accuracy_score",acc_score)
    print("precision_score",prec_score)
    print("rec_score",rec_score)


neuralnet_clf_sklearn()
