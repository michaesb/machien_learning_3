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
    ratio_ = 0.1
    X_train, X_test, y_train, y_test = retrieve_data(undersampling = True, ratio= ratio_)


    learning_rate = 10**(-np.linspace(0, 6, 70))
    n = len(learning_rate)
    timer = np.zeros(n)
    acc_score = np.zeros(n)
    rec_score = np.zeros(n)
    prec_score = np.zeros(n)
    for i in range(len(learning_rate)):
        time1 = time.time()
        print(int(100*i/len(learning_rate)), "%", end = "\r")
        clf = sklearn.neural_network.MLPClassifier(
                                hidden_layer_sizes = (20,20),
                                learning_rate = "adaptive",
                                learning_rate_init = learning_rate[i],
                                max_iter = 1000,
                                tol = 1e-10,
                                verbose = False,
                                )
        clf = clf.fit(X_train, y_train.ravel())
        predict = clf.predict(X_test)
        acc_score[i] = accuracy_score(y_test.ravel(),predict)
        prec_score[i] = precision_score(y_test.ravel(),predict)
        rec_score[i] = recall_score(y_test.ravel(),predict)
        time2 = time.time()
        timer[i] =time2 -time1
        # print("time = ",timer[i]," s")

    plt.semilogx(learning_rate,acc_score)
    plt.semilogx(learning_rate,prec_score)
    plt.semilogx(learning_rate,rec_score)
    plt.legend(['accuracy', "precision", "recall"])
    plt.xlabel(r"Learning rate $\eta$")
    plt.ylabel("Accuracy score")
    plt.title("Scikit-Learn NeuralNet score for different learning rates")
    plt.show()
    print("ratio: ", ratio_)
    print("accuracy_score",acc_score)
    print("precision_score",prec_score)
    print("rec_score",rec_score)


neuralnet_clf_sklearn()
