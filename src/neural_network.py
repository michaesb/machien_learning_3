#!/usr/bin/env python
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer, confusion_matrix
from sklearn.model_selection import GridSearchCV
from data_process import retrieve_data
import matplotlib.pyplot as plt
import sklearn.neural_network
import scikitplot as skplt
from scoring import scores
import sklearn.metrics
import seaborn as sb
import numpy as np



"""
runs the classifier class with data from the data package
"""

def learningrate_nn():
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
                                hidden_layer_sizes = (100,100,100,100,100,100),
                                learning_rate = "adaptive",
                                learning_rate_init = learning_rate[i],
                                max_iter = 1000000,
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



def grid_search_nn():
    X_train, X_test, y_train, y_test = retrieve_data( undersampling=True, ratio=1.0, random_state=3 )

    clf = sklearn.neural_network.MLPClassifier(
                                learning_rate = "adaptive",
                                learning_rate_init = 0.001,
                                tol = 1e-4,
                                verbose = False
                                )

    ## Grid search
    param_grid = {
        "hidden_layer_sizes" : [(30),(40,40),(50,50,50),(30,30,30,30)],
        "activation": ["logistic"],
        "solver": ["lbfgs","adam"],
        "alpha": [0.1, 0.01, 0.001],
        "max_iter": [500,1000]
    }
    
    # Best params for Recall Score {'activation': 'logistic', 'alpha': 0.1, 'hidden_layer_sizes': (30, 30, 30, 30), 'max_iter': 500, 'solver': 'lbfgs'}
    # Accuracy Test Score:   0.9231
    # Precision Test Score:  0.9062. What percentage of the predicted frauds were frauds?
    # Recall Test Score:     0.9355. What percentage of the actual frauds were predicted?
    # Recall Train Score     1.0000
    # Recall CV Train Score: 1.0000
    # Recall CV Test Score:  0.9555

    scorers = {
        "precision_score": make_scorer(precision_score),
        "recall_score": make_scorer(recall_score),
        "accuracy_score": make_scorer(accuracy_score)
    }
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring=scorers, refit="recall_score", return_train_score=True, n_jobs=-1)
    
    grid_search.fit(X_train, y_train)
    prediction = grid_search.predict(X_test)

    scores(prediction, y_test, X_train, y_train, grid_search)


def neuralnet():
    X_train, X_test, y_train, y_test = retrieve_data( undersampling=True, ratio=1.0, random_state=3 )

    clf = sklearn.neural_network.MLPClassifier(
                                learning_rate = "adaptive",
                                learning_rate_init = 0.001,
                                activation= "logistic",
                                alpha = 0.1,
                                hidden_layer_sizes = (30, 30, 30, 30),
                                max_iter = 500,
                                solver = "lbfgs",
                                tol = 1e-4,
                                verbose = False
                                )


    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    scores(prediction, y_test, X_train, y_train)
    

# learningrate_nn()
grid_search_nn()
# neuralnet()
