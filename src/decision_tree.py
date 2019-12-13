#!/usr/bin/env python
from sklearn.metrics import accuracy_score, recall_score, precision_score, make_scorer
from sklearn.model_selection import GridSearchCV
from data_process import retrieve_data
from plotting import plotting_ratio
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn import tree
import numpy as np
import time

"""
runs the classifier class with data from the data package
"""

def decisionsTree_clf_sklearn():
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

# decisionsTree_clf_sklearn()


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


    # Best params for Recall Score {'criterion': 'gini', 'max_depth': 10, 'max_features': 25, 'min_samples_leaf': 1, 'min_samples_split': 2}
    # test score()  0.9375
    # train score() 1.0
    # Accuracy Score:  0.9200
    # Precision Score: 0.9036. What percentage of the predicted frauds were frauds?
    # Recall Score:    0.9375. What percentage of the actual frauds were predicted?
        
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

    # tree.export_graphviz(grid_search.best_estimator_, out_file="tree.dot")

grid_search_decisiontree()