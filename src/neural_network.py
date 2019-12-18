#!/usr/bin/env python

## Importing needed packages and methods
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import sklearn.neural_network
import sklearn.metrics
import numpy as np

## Importing own helper funcions
from data_process import retrieve_data
from scoring import scores



def neuralnet_learningrate():
    """
    This function tests using a neural network with different,
    initial learning rate and then plots and prints the results.
    """

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
                                hidden_layer_sizes = (30, 30, 30, 30),
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

    plt.semilogx(learning_rate, acc_score)
    plt.semilogx(learning_rate, prec_score)
    plt.semilogx(learning_rate, rec_score)
    plt.legend(["Accuracy", "Precision", "Recall"], prop={'size': 12})
    plt.xlabel(r"Learning rate $\eta$", size=14)
    plt.ylabel("Scores", size=14)
    plt.title("Scikit-Learn NeuralNet score for different learning rates", size=16)
    plt.show()
    print("Ratio: ",ratio_)
    print("Accuracy score",acc_score)
    print("Precision score",prec_score)
    print("Recall score",rec_score)


def neuralnet_gridsearch():
    """
    This function retrieves the dataset, creates a neural network Multilayered Perceptron, and
    uses a grid search method to find the most optimum parameters for maximizing the recall score.
    """

    X_train, X_test, y_train, y_test = retrieve_data( undersampling=True, ratio=1.0, random_state=3 )

    ## We decided on using the adaptive learning rate and a inital rate of 0.001.
    clf = sklearn.neural_network.MLPClassifier(
                                learning_rate = "adaptive",
                                learning_rate_init = 0.001,
                                tol = 1e-4,
                                verbose = False
                                )

    ## Grid search parameter grid to search through.
    param_grid = {
        "hidden_layer_sizes" : [(30),(40,40),(50,50,50),(30,30,30,30)],
        "activation": ["logistic"],
        "solver": ["lbfgs","adam"],
        "alpha": [0.1, 0.01, 0.001],
        "max_iter": [500,1000]
    }

    ## Different scorers for the grid search
    scorers = {
        "precision_score": make_scorer(precision_score),
        "recall_score": make_scorer(recall_score),
        "accuracy_score": make_scorer(accuracy_score)
    }

    ## Creating the grid search object. Using refit="recall_score" to optimize using this score
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring=scorers, refit="recall_score", return_train_score=True, n_jobs=-1)

    grid_search.fit(X_train, y_train)
    prediction = grid_search.predict(X_test)

    scores(prediction, y_test, X_train, y_train, grid_search)


def neuralnet_tuned():
    """
    This function reads the dataset and uses the neural network ..
    to test optimized parameters found in the function above.
    """

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



### Uncomment the function you'd like to run:

# neuralnet_learningrate()
neuralnet_gridsearch()
# neuralnet_tuned()
