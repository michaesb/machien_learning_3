#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
from scipy.integrate import simps

import time
import sklearn.neural_network
import sklearn.metrics
from data_process import retrieve_data
from sklearn.model_selection import train_test_split

"""
runs the classifier class with data from the data package
"""

def neuralnet_clf_sklearn():

    X,y = retrieve_data()
    # print ("ohh, fuck" )
    # print (X)
    # print ("I can't believe done this")
    # print (np.sum(y)/len(y)*100)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4)
    learning_rate = np.linspace(0.01, 0.001, 11)
    n = len(learning_rate)
    timer = np.zeros(n)
    accuracy_score = np.zeros(n)
    for i in range(len(learning_rate)):
        time1 = time.time()
        print(int(100*i/len(learning_rate)), "%")
        reg = sklearn.neural_network.MLPClassifier(
                                hidden_layer_sizes = (1),
                                learning_rate = "adaptive",
                                learning_rate_init = learning_rate[i],
                                max_iter = 1000,
                                tol = 1e-10,
                                verbose = True,
                                )
        reg = reg.fit(X_train, y_train.ravel())
        predict = reg.predict(X_test)
        accuracy_score[i] = reg.score(X_test,y_test.ravel())
        time2 = time.time()
        timer[i] =time2 -time1
        # print("time = ",timer[i]," s")
    plt.semilogx(learning_rate,accuracy_score, "*")
    plt.semilogx(learning_rate,accuracy_score)
    plt.xlabel(r"Learning rate $\eta$")
    plt.ylabel("Accuracy score")
    plt.title("Scikit-Learn NeuralNet score for different learning rates")
    plt.show()

    print(accuracy_score)

neuralnet_clf_sklearn()
