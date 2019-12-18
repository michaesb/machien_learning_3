#!/usr/bin/env python
from sklearn.metrics import accuracy_score, recall_score, precision_score, make_scorer, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from data_process import retrieve_data
import matplotlib.pyplot as plt
import scikitplot as skplt
from scoring import scores
import seaborn as sb
import numpy as np


def grid_search_logreg():

    X_train, X_test, y_train, y_test = retrieve_data( undersampling=True, ratio=1.0, random_state=3 )
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.append(y_train, y_test)


    ### Logistic Regression
    clf = LogisticRegression(random_state=4, solver="liblinear")

    ## Grid search
    param_grid= {
        "C" : np.logspace(-3,3,7),
        "penalty" : ["l1", "l2"]
    }

    scorers = {
        "precision_score": make_scorer(precision_score),
        "recall_score": make_scorer(recall_score),
        "accuracy_score": make_scorer(accuracy_score)
    }
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring=scorers, refit="recall_score", return_train_score=True, n_jobs=-1)

    grid_search.fit(X_train, y_train)
    prediction = grid_search.predict(X_test)


    scores(prediction, y_test, X_train, y_train, grid_search)



def logreg():
    X_train, X_test, y_train, y_test = retrieve_data( undersampling=True, ratio=1.0, random_state=3 )
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.append(y_train, y_test)

    ### Logistic Regression
    clf = LogisticRegression(random_state=4, solver="liblinear", C=0.01, penalty="l1")

    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    scores(prediction, y_test, X_train, y_train)


grid_search_logreg()
# logreg()
