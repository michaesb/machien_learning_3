#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
import time
from sklearn import tree
from sklearn.metrics import accuracy_score, recall_score, precision_score
from plotting import plotting_ratio
from data_process import retrieve_data

"""
runs the classifier class with data from the data package
"""

def decisionsTree_clf_sklearn():
    # ratio_ = 0.25

    n = 61
    ratio_ = 10**(-np.linspace(0.5, 0.0, n))
    # ratio_ = np.arange(1,101)/100
    print("ratio: ", ratio_)
    n = len(ratio_)
    timer = np.zeros(n)
    acc_score = np.zeros(n)
    rec_score = np.zeros(n)
    prec_score = np.zeros(n)
    for i in range(n):
        X_train, X_test, y_train, y_test = retrieve_data(undersampling = True, ratio = ratio_[i])
        time1 = time.time()
        print(int(100*i/len(ratio_)), "%", end= "\r")

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train.ravel())
        predict = clf.predict(X_test)
        acc_score[i] = accuracy_score(y_test.ravel(),predict)
        prec_score[i] = precision_score(y_test.ravel(),predict)
        rec_score[i] = recall_score(y_test.ravel(),predict)
        time2 = time.time()
        timer[i] =time2 -time1
        # print("time = ",timer[i]," s")

    plotting_ratio(ratio_, acc_score, prec_score, rec_score, "DecisionsTree")
    print("accuracy_score",acc_score)
    print("precision_score",prec_score)
    print("rec_score",rec_score)

decisionsTree_clf_sklearn()
