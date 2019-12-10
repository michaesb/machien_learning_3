#!/usr/bin/env python
from sklearn.metrics import accuracy_score, recall_score, precision_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from data_process import retrieve_data
import matplotlib.pyplot as plt
import scikitplot as skplt
import numpy as np

X_train, X_test, y_train, y_test = retrieve_data( undersampling=True, ratio=1, random_state=None)

### Random Forest Classifier
clf = RandomForestClassifier(random_state=4)

## Grid search
param_grid = {
    "n_estimators" : [10, 100, 200],
    "min_samples_split": [3, 5, 10, 15], 
    "max_depth": [3, 5, 15, 25],
    "max_features": [5, 20, 30, "auto", "sqrt", "log2"]
}

# Best params for Recall Score {'max_depth': 15, 'max_features': 30, 'min_samples_split': 15, 'n_estimators': 10}
# test score()  0.891566265060241
# train score() 0.9631901840490797
# Accuracy Score:  0.9231
# Precision Score: 0.9548. What percentage of the predicted frauds were frauds?
# Recall Score:    0.8916. What percentage of the actual frauds were predicted?


# scorers = {
#     "precision_score": make_scorer(precision_score),
#     "recall_score": make_scorer(recall_score),
#     "accuracy_score": make_scorer(accuracy_score)
# }
# grid_search = GridSearchCV(clf, param_grid, cv=5, scoring=scorers, refit="recall_score", return_train_score=True, n_jobs=-1)


grid_search = GridSearchCV(clf, param_grid, cv=5, scoring="recall", n_jobs=-1)
grid_search.fit(X_train, y_train)

prediction = grid_search.predict(X_test)
print("Best params for Recall Score", grid_search.best_params_)

# print("best_score_  ",grid_search.best_score_) ## USELESS. RETURNS THE BEST SCORE FOR A PARTICULAR FOLD
print("test score() ",grid_search.score(X_test, y_test))
print("train score()",grid_search.score(X_train, y_train))

# clf = RandomForestClassifier(random_state=4, **grid_search.best_params_)
# clf.fit(X_train, y_train)
# prediction = clf.predict(X_test)

acc = accuracy_score(y_test, prediction)
print(f"Accuracy Score:  {acc:.4f}")
precision = precision_score(y_test, prediction)
print(f"Precision Score: {precision:.4f}. What percentage of the predicted frauds were frauds?" )
recall = recall_score(y_test, prediction)
print(f"Recall Score:    {recall:.4f}. What percentage of the actual frauds were predicted?")