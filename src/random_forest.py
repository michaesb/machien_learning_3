#!/usr/bin/env python
from sklearn.metrics import accuracy_score, recall_score, precision_score, make_scorer, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from data_process import retrieve_data
import matplotlib.pyplot as plt
import scikitplot as skplt
from scoring import scores
import seaborn as sb
import numpy as np

def grid_search_randomforest():
    X_train, X_test, y_train, y_test = retrieve_data( undersampling=True, ratio=1.0, random_state=None)

    ### Random Forest Classifier
    clf = RandomForestClassifier(random_state=4)

    ## Grid search
    param_grid = {
        "n_estimators" : [10, 100, 200],
        "min_samples_split": [3, 5, 10, 15], 
        "max_depth": [3, 5, 15, 25],
        "max_features": [5, 20, 30, "auto", "sqrt", "log2"]
    }

    # Best params for Recall Score {'max_depth': 15, 'max_features': 30, 'min_samples_split': 5, 'n_estimators': 10}
    # Accuracy Test Score:   0.9169
    # Precision Test Score:  0.9448. What percentage of the predicted frauds were frauds?
    # Recall Test Score:     0.8953. What percentage of the actual frauds were predicted?
    # Recall Train Score     0.9844
    # Recall CV Train Score: 1.0000
    # Recall CV Test Score:  0.9156

    # ratio 0.01
    # Best params for Recall Score {'max_depth': 25, 'max_features': 5, 'min_samples_split': 5, 'n_estimators': 200}
    # Accuracy Test Score:   0.9975
    # Precision Test Score:  0.9577. What percentage of the predicted frauds were frauds?
    # Recall Test Score:     0.7953. What percentage of the actual frauds were predicted?
    # Recall Train Score     0.9502
    # Recall CV Train Score: 1.0000
    # Recall CV Test Score:  0.8537


    scorers = {
        "precision_score": make_scorer(precision_score),
        "recall_score": make_scorer(recall_score),
        "accuracy_score": make_scorer(accuracy_score)
    }
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring=scorers, refit="recall_score", return_train_score=True, n_jobs=-1)

    grid_search.fit(X_train, y_train)
    prediction = grid_search.predict(X_test)

    scores(prediction, y_test, X_train, y_train, grid_search)

def randomforest():
    X_train, X_test, y_train, y_test = retrieve_data( undersampling=True, ratio=1, random_state=None)

    ### Random Forest Classifier
    clf = RandomForestClassifier(
                                max_depth = 15,
                                max_features = 30,
                                min_samples_split = 5, 
                                n_estimators = 10
                                )

    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    scores(prediction, y_test, X_train, y_train)
    

grid_search_randomforest()
# randomforest()