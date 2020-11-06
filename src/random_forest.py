#!/usr/bin/env python

## Importing needed packages and methods
from sklearn.metrics import accuracy_score, recall_score, precision_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

## Importing own helper funcions
from data_process import retrieve_data
from scoring import scores



def randomforest_gridsearch():
    """
    This function retrieves the dataset and uses a random forest classifier for predicting
    credit card frauds. To maximize the recall score, we used a grid search method to optimize
    the parameters going into the random forest classifier.
    """

    X_train, X_test, y_train, y_test = retrieve_data( undersampling=True, ratio=1.0, random_state=None)

    ### Random Forest Classifier
    clf = RandomForestClassifier(random_state=4)

    ## Grid search parameter grid to search through
    param_grid = {
        "criterion": ["gini","entropy"],
        "n_estimators" : [10, 100, 200],
        "min_samples_split": [3, 5, 10],
        "max_depth": [5, 15, 25],
        "max_features": [5, 10, 30],
        "min_samples_leaf": [1, 10, 20]
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



def randomforest_tuned():
    """
    This function reads the dataset and uses the random forest ..
    to test optimized parameters found in the function above.
    """

    X_train, X_test, y_train, y_test = retrieve_data( undersampling=True, ratio=1, random_state=None)
    print("shape of X_train " +str(np.shape(X_train)))
    print("shape of Y_train " +str(np.shape(y_train)))
    print("shape of X_test " +str(np.shape(X_test)))
    print("shape of Y_test " +str(np.shape(y_test)))
    clf = RandomForestClassifier()


    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    scores(prediction, y_test, X_train, y_train)



### Uncomment the function you'd like to run:

randomforest_gridsearch()
#randomforest_tuned()
