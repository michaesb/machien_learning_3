from data_process import retrieve_data
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = retrieve_data( undersampling=True )

### Logistic Regression
logreg = LogisticRegression(random_state=4,
                            solver='lbfgs',
                            multi_class='multinomial',
                            max_iter=1000)


param_grid= {"C":np.logspace(-3,3,7), "penalty":["l2"]}
logreg_grid = GridSearchCV(logreg, param_grid, cv=5)
# logreg.fit(X_train, y_train)
logreg_grid.fit(X_train, y_train)

# prediction = logreg.predict(X_test)
prediction = logreg_grid.predict(X_test)

acc = accuracy_score(y_test, prediction)
print(f"Accuracy Score:  {acc:.4f}")
precision = precision_score(y_test, prediction)
print(f"Precision Score: {precision:.4f}. What percentage of the predicted frauds were frauds?" )
recall = recall_score(y_test, prediction)
print(f"Recall Score:    {recall:.4f}. What percentage of the actual frauds were predicted?")
