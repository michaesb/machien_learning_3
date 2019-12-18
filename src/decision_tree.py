#!/usr/bin/env python

## Importing needed packages and methods
from sklearn.metrics import accuracy_score, recall_score, precision_score, make_scorer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np
import graphviz

## Importing own helper funcions
from data_process import retrieve_data
from scoring import scores



def decisiontree_undersamplingratio():
    """
    This funcions purpose is to test how the scores vary when using different
    undersampling ratios and plots the results.
    """

    n = 61

    ratio_ = 10**(-np.linspace(6.0, 0.0, n))
    n = len(ratio_)
    acc_score = np.zeros(n)
    rec_score = np.zeros(n)
    prec_score = np.zeros(n)
    for i in range(n):

        print(int(100*i/len(ratio_)), "%", end= "\r")
        X_train, X_test, y_train, y_test = retrieve_data(undersampling = True, ratio = ratio_[i])

        clf = tree.DecisionTreeClassifier()

        clf = clf.fit(X_train, y_train.ravel())
        predict = clf.predict(X_test)

        acc_score[i] = accuracy_score(y_test.ravel(), predict)
        prec_score[i] = precision_score(y_test.ravel(), predict)
        rec_score[i] = recall_score(y_test.ravel(), predict)

    plt.semilogx(ratio_, acc_score)
    plt.semilogx(ratio_, prec_score)
    plt.semilogx(ratio_, rec_score)
    plt.xlabel("Ratio", size=14)
    plt.ylabel("Score", size=14)
    plt.title("Scikit-Learn Decision Tree score for different ratios", size=16)
    plt.legend(['Accuracy', "Precision", "Recall"], prop={'size': 12})
    plt.savefig("plots/dectree_ratiotest.png")
    plt.show()


def decisiontree_gridsearch():
    """
    This function retrieves the dataset and uses grid search to find optimum parameters
    to optimize the recall score of a Decision Tree classifier.
    """

    X_train, X_test, y_train, y_test = retrieve_data( undersampling=True, ratio=1.0, random_state=2 )

    ## Our decision tree model
    clf = tree.DecisionTreeClassifier()

    # Grid search parameter grid to search through.
    param_grid = {
        "criterion": ["gini","entropy"],
        "min_samples_split": [2, 3, 5, 8, 10], 
        "max_depth": [3, 5, 10, 15, 20, 25],
        "max_features": [5, 20, 25, 30, "auto", "sqrt", "log2"],
        "min_samples_leaf": [1, 5, 10, 20, 50]
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

    ## Using the graphviz package to produce a PNG image to display the decision tree
    dot_data = tree.export_graphviz(grid_search.best_estimator_, out_file=None, filled=True, rounded=True,  special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.format = "png"
    graph.render("plots/tree")


    
def decisiontree_tuned():
    """
    This function reads the dataset and uses the decision tree ..
    to test optimized parameters found in the function above.
    """

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
    graph.render("plots/tree")



### Uncomment the function you'd like to run:

# decisiontree_undersamplingratio()
# decisiontree_gridsearch()
decisiontree_tuned()

