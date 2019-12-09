from sklearn.metrics import accuracy_score, recall_score, precision_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from data_process import retrieve_data
import matplotlib.pyplot as plt
import scikitplot as skplt
import numpy as np

X_train, X_test, y_train, y_test = retrieve_data( undersampling=True, ratio=1 )

### Logistic Regression
logreg = LogisticRegression(random_state=4, solver='liblinear')

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
logreg_grid = GridSearchCV(logreg, param_grid, cv=5, scoring=scorers, refit="recall_score", n_jobs=-1)
logreg_grid.fit(X_train, y_train)

prediction = logreg_grid.predict(X_test)
print("Best params for Recall Score", logreg_grid.best_params_)


# logreg.fit(X_train, y_train)
# prediction = logreg.predict(X_test)

acc = accuracy_score(y_test, prediction)
print(f"Accuracy Score:  {acc:.4f}")
precision = precision_score(y_test, prediction)
print(f"Precision Score: {precision:.4f}. What percentage of the predicted frauds were frauds?" )
recall = recall_score(y_test, prediction)
print(f"Recall Score:    {recall:.4f}. What percentage of the actual frauds were predicted?")