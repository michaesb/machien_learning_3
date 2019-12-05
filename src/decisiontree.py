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

    X_train, X_test, y_train, y_test = retrieve_data(undersampling = True)


    learning_rate = np.linspace(0.01, 0.001, 11)
    n = len(learning_rate)
    timer = np.zeros(n)
    accuracy_score = np.zeros(n)
    for i in range(len(learning_rate)):
        time1 = time.time()
        print(int(100*i/len(learning_rate)), "%")
        clf = tree.DecisionTreeClassifier(random_state = 0)
        clf = clf.fit(X_train, y_train.ravel())
        predict = clf.predict(X_test)
        accuracy_score[i] = clf.score(X_test,y_test.ravel())
        time2 = time.time()
        timer[i] =time2 -time1
        # print("time = ",timer[i]," s")

    print(precision_score)
    print( )

neuralnet_clf_sklearn()
