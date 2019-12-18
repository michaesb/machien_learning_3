#!/usr/bin/env python

## Importing needed packages
from sklearn.metrics import accuracy_score, recall_score, precision_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np

## Importing helper functions
from data_process import retrieve_data
from scoring import scores


def logreg_gridsearch():
    """
    This function reads in the dataset and performs a logistic regression method and .. 
    using Grid Search to find the optimum parameters that will maximize the recall score,
    """

    X_train, X_test, y_train, y_test = retrieve_data( undersampling=True, ratio=1.0, random_state=3 )

    clf = LogisticRegression(random_state=4, solver="liblinear")

    ## Grid search parameter grid
    param_grid= {
        "C" : np.logspace(-3,3,7),
        "penalty" : ["l1", "l2"]
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



def logreg_tuned():
    """
    This function reads the dataset and uses logistic regression ..
    to test optimized parameters found in the function above.
    """

    X_train, X_test, y_train, y_test = retrieve_data( undersampling=True, ratio=1.0, random_state=3 )

    clf = LogisticRegression(random_state=4, solver="liblinear", C=0.01, penalty="l1")

    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    scores(prediction, y_test, X_train, y_train)


## Uncomment the function you'd like to run:

logreg_gridsearch()
# logreg_tuned()
