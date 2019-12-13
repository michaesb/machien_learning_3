#!/usr/bin/env python
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import GridSearchCV
from data_process import retrieve_data
# from scipy.integrate import simps
import matplotlib.pyplot as plt
import sklearn.neural_network
import scikitplot as skplt
import sklearn.metrics
import numpy as np



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


# neuralnet_clf_sklearn()

def grid_search_nn():
    X_train, X_test, y_train, y_test = retrieve_data( undersampling=True, ratio=1, random_state=3 )

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
    # param_grid = {
    #     # "hidden_layer_sizes" : [(10,10,10,10), (20,20,20,20), (10,10,10,10,10)],
    #     "hidden_layer_sizes" : [(30),(40,40),(50,50,50),(30,30,30,30), (30,30,30,30,30)],
    #     "activation": ["logistic", "tanh"],g
    #     "solver": ["lbfgs","adam", "sgd"],
    #     "alpha": [0.1, 0.01, 0.001, 0.0001],
    #     "max_iter": [500,1000, 20000]
    # }

    # Best params for Recall Score {'activation': 'logistic', 'alpha': 0.01, 'hidden_layer_sizes': (30, 30, 30, 30), 'max_iter': 1000, 'solver': 'adam'}    
    # Accuracy Score:  0.9477
    # Precision Score: 0.9422. What percentage of the predicted frauds were frauds?
    # Recall Score:    0.9588. What percentage of the actual frauds were predicted?


    scorers = {
        "precision_score": make_scorer(precision_score),
        "recall_score": make_scorer(recall_score),
        "accuracy_score": make_scorer(accuracy_score)
    }
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring=scorers, refit="recall_score", return_train_score=True, n_jobs=-1)
    # grid_search = GridSearchCV(clf, param_grid, cv=5, scoring="recall", n_jobs=-1)
    
    grid_search.fit(X_train, y_train)
    prediction = grid_search.predict(X_test)


    print("Best params for Recall Score", grid_search.best_params_)

    acc = accuracy_score(y_test, prediction)
    print(f"Accuracy Test Score:   {acc:.4f}")
    precision = precision_score(y_test, prediction)
    print(f"Precision Test Score:  {precision:.4f}. What percentage of the predicted frauds were frauds?" )
    recall = recall_score(y_test, prediction)
    print(f"Recall Test Score:     {recall:.4f}. What percentage of the actual frauds were predicted?")
    print(f"Recall Train Score     {grid_search.score(X_train, y_train):.4f}")


    mean_train_recall_score    = grid_search.cv_results_["mean_train_recall_score"]
    index = np.argmax( mean_train_recall_score )
    print(f"Recall CV Train Score: {mean_train_recall_score[index]:.4f}" )
    mean_test_recall_score    = grid_search.cv_results_["mean_test_recall_score"]
    index = np.argmax( mean_test_recall_score )
    print(f"Recall CV Test Score:  {mean_test_recall_score[index]:.4f}" )


grid_search_nn()
