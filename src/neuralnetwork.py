#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
from scipy.integrate import simps
import time
import sklearn.neural_network
import sklearn.metrics
from data_process import retrieve_data
from sklearn.metrics import accuracy_score, recall_score, precision_score


"""
runs the classifier class with data from the data package
"""

def neuralnet_clf_sklearn():

    X_train, X_test, y_train, y_test = retrieve_data(undersampling = 0)


    learning_rate = np.linspace(0.01, 0.001, 11)
    n = len(learning_rate)
    timer = np.zeros(n)
    acc_score = np.zeros(n)
    rec_score = np.zeros(n)
    prec_score = np.zeros(n)

    for i in range(len(learning_rate)):
        time1 = time.time()
        print(int(100*i/len(learning_rate)), "%")
        clf = sklearn.neural_network.MLPClassifier(
                                hidden_layer_sizes = (1),
                                learning_rate = "adaptive",
                                learning_rate_init = learning_rate[i],
                                max_iter = 1000,
                                tol = 1e-10,
                                verbose = True,
                                )
        clf = clf.fit(X_train, y_train.ravel())
        predict = clf.predict(X_test)
        acc_score[i] = accuracy_score(y_test.ravel(),predict)
        prec_score[i] = precision_score(y_test.ravel(),predict)
        rec_score[i] = recall_score(y_test.ravel(),predict)
        time2 = time.time()
        timer[i] =time2 -time1
        # print("time = ",timer[i]," s")
    plt.semilogx(learning_rate,acc_score, "*")
    plt.semilogx(learning_rate,acc_score)
    plt.xlabel(r"Learning rate $\eta$")
    plt.ylabel("Accuracy score")
    plt.title("Scikit-Learn NeuralNet score for different learning rates")
    plt.show()
    print("ohh, fuck ")
    print(acc_score)
    print(prec_score)
    print(rec_score)
    print ("I can't believe you have done this")

neuralnet_clf_sklearn()
