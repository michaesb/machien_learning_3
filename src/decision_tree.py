#!/usr/bin/env python
from sklearn.metrics import accuracy_score, recall_score, precision_score, make_scorer, confusion_matrix
from sklearn.model_selection import GridSearchCV
from data_process import retrieve_data
from plotting import plotting_ratio
import matplotlib.pyplot as plt
import scikitplot as skplt
from scoring import scores
from sklearn import tree
import seaborn as sb
import numpy as np
import graphviz
import time

"""
runs the classifier class with data from the data package
"""

def ratio_decisiontree():
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

        # import scikitplot as skplt
        # skplt.metrics.plot_confusion_matrix(y_test, predict, normalize=True, hide_counts=False)
        # plt.show()

    plotting_ratio(ratio_, acc_score, prec_score, rec_score, "DecisionsTree")
    print("accuracy_score",acc_score)
    print("precision_score",prec_score)
    print("rec_score",rec_score)



def grid_search_decisiontree():

    X_train, X_test, y_train, y_test = retrieve_data( undersampling=True, ratio=1, random_state=2 )

    clf = tree.DecisionTreeClassifier()

    # Grid search
    param_grid = {
        "criterion": ["gini","entropy"],
        "min_samples_split": [2, 3, 5, 8, 10], 
        "max_depth": [3, 5, 10, 15, 20, 25],
        "max_features": [5, 20, 25, 30, "auto", "sqrt", "log2"],
        "min_samples_leaf": [1, 5, 10, 20, 50]
    }


    # Best params for Recall Score {'criterion': 'gini', 'max_depth': 20, 'max_features': 30, 'min_samples_leaf': 1, 'min_samples_split': 2}
    # Accuracy Test Score:   0.9108
    # Precision Test Score:  0.8970. What percentage of the predicted frauds were frauds?
    # Recall Test Score:     0.9250. What percentage of the actual frauds were predicted?
    # Recall Train Score     1.0000
    # Recall CV Train Score: 1.0000
    # Recall CV Test Score:  0.9367

    scorers = {
        "precision_score": make_scorer(precision_score),
        "recall_score": make_scorer(recall_score),
        "accuracy_score": make_scorer(accuracy_score)
    }
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring=scorers, refit="recall_score", return_train_score=True, n_jobs=-1)
    
    grid_search.fit(X_train, y_train)
    prediction = grid_search.predict(X_test)

    scores(prediction, y_test, X_train, y_train, grid_search)

    dot_data = tree.export_graphviz(grid_search.best_estimator_, out_file=None, filled=True, rounded=True,  special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.format = "png"
    graph.render("tree")

    
def decisiontree():

    X_train, X_test, y_train, y_test = retrieve_data( undersampling=True, ratio=1, random_state=2 )

    clf = tree.DecisionTreeClassifier(
                                    criterion = "gini",
                                    max_depth = 20,
                                    max_features = 30,
                                    min_samples_leaf = 1,
                                    min_samples_split = 2
                                    )


    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    scores(prediction, y_test, X_train, y_train)

    dot_data = tree.export_graphviz(clf, out_file=None, filled=True, rounded=True,  special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.format = "png"
    graph.render("tree")

# ratio_decisiontree()

grid_search_decisiontree()

# decisiontree()

