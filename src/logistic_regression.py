#!/usr/bin/env python
from sklearn.metrics import accuracy_score, recall_score, precision_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from data_process import retrieve_data
import matplotlib.pyplot as plt
import scikitplot as skplt
import numpy as np

X_train, X_test, y_train, y_test = retrieve_data( undersampling=True, ratio=1, random_state=3 )
X = np.concatenate((X_train, X_test), axis=0)
y = np.append(y_train, y_test)


### Logistic Regression
clf = LogisticRegression(random_state=4, solver='liblinear')

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
# grid_search = GridSearchCV(clf, param_grid, cv=5, scoring="recall", n_jobs=-1)

grid_search.fit(X_train, y_train)
exit()
prediction = grid_search.predict(X_test)


print("Best params for Recall Score", grid_search.best_params_)

acc = accuracy_score(y_test, prediction)
print(f"Accuracy Test Score:   {acc:.4f}")
precision = precision_score(y_test, prediction)
print(f"Precision Test Score:  {precision:.4f}. What percentage of the predicted frauds were frauds?" )
recall = recall_score(y_test, prediction)
print(f"Recall Test Score:     {recall:.4f}. What percentage of the actual frauds were predicted?")
print(f"Recall Train Score     {grid_search.score(X_train, y_train):.4f}")

# print(grid_search.cv_results_)

mean_train_recall_score    = grid_search.cv_results_["mean_train_recall_score"]
# mean_train_precision_score = grid_search.cv_results_["mean_train_precision_score"]
# mean_train_accuracy_score  = grid_search.cv_results_["mean_train_accuracy_score"]
index = np.argmax( mean_train_recall_score )
print(f"Recall CV Train Score: {mean_train_recall_score[index]:.4f}" )
# print( mean_train_precision_score[index] )
# print( mean_train_accuracy_score[index] )

mean_test_recall_score    = grid_search.cv_results_["mean_test_recall_score"]
# mean_test_precision_score = grid_search.cv_results_["mean_test_precision_score"]
# mean_test_accuracy_score  = grid_search.cv_results_["mean_test_accuracy_score"]
index = np.argmax( mean_test_recall_score )
print(f"Recall CV Test Score:  {mean_test_recall_score[index]:.4f}" )
# print( mean_test_precision_score[index] )
# print( mean_test_accuracy_score[index] )

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
ax= plt.subplot()
cm = confusion_matrix(y_test, prediction)
sns.heatmap(cm, annot=True, ax = ax,fmt ="g"); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Fraud', 'Non-fraud'])
ax.yaxis.set_ticklabels(['Fraud', 'Non-fraud'])
plt.show()
