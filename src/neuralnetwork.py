#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
from scipy.integrate import simps
import sklearn.neural_network
import sklearn.metrics
from data_process import retrieve_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer

from sklearn.model_selection import GridSearchCV


"""
runs the classifier class with data from the data package
"""

def neuralnet_clf_sklearn():
    ratio_ = 0.1
    X_train, X_test, y_train, y_test = retrieve_data(undersampling = True, ratio= ratio_)


    learning_rate = 10**(-np.linspace(3, 1, 70))
    n = len(learning_rate)
    acc_score = np.zeros(n)
    rec_score = np.zeros(n)
    prec_score = np.zeros(n)
    for i in range(len(learning_rate)):

        print(int(100*i/len(learning_rate)), "%", end = "\r")
        clf = sklearn.neural_network.MLPClassifier(
                                hidden_layer_sizes = (80,70,60,50,40),
                                learning_rate = "adaptive",
                                learning_rate_init = learning_rate[i],
                                max_iter = 10000,
                                tol = 1e-10,
                                verbose = False,
                                )
        clf = clf.fit(X_train, y_train.ravel())
        predict = clf.predict(X_test)
        acc_score[i] = accuracy_score(y_test.ravel(),predict)
        prec_score[i] = precision_score(y_test.ravel(),predict)
        rec_score[i] = recall_score(y_test.ravel(),predict)




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

def grid_search_nn():
    X_train, X_test, y_train, y_test = retrieve_data( undersampling=True, ratio=1 )

    clf = sklearn.neural_network.MLPClassifier(
                                hidden_layer_sizes = (10,10),
                                learning_rate = "adaptive",
                                learning_rate_init = 0.001,
                                max_iter = 1000,
                                tol = 1e-10,
                                verbose = False,
                                )

    ## Grid search
    param_grid = {
        "hiddel_layer_sizes" : [(10),(10,10),(10,10,10),(10,10,10,10)],
        "learning_rate_init": [0.1, 0.01, 0.001, 0.0001]
    }

    scorers = {
        "precision_score": make_scorer(precision_score),
        "recall_score": make_scorer(recall_score),
        "accuracy_score": make_scorer(accuracy_score)
    }

    nn_grid = GridSearchCV(clf, param_grid, cv=5, scoring=scorers, refit="recall_score", n_jobs=-1)
    nn_grid.fit(X_train, y_train)

    prediction = nn_grid.predict(X_test)
    print("Best params for Recall Score", nn_grid.best_params_)


    # logreg.fit(X_train, y_train)
    # prediction = logreg.predict(X_test)

    acc = accuracy_score(y_test, prediction)
    print(f"Accuracy Score:  {acc:.4f}")
    precision = precision_score(y_test, prediction)
    print(f"Precision Score: {precision:.4f}. What percentage of the predicted frauds were frauds?" )
    recall = recall_score(y_test, prediction)
    print(f"Recall Score:    {recall:.4f}. What percentage of the actual frauds were predicted?")
