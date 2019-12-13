from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

def scores(clf, prediction, y_test, X_train, y_train):
    print("Best params for Recall Score", clf.best_params_)

    acc = accuracy_score(y_test, prediction)
    print(f"Accuracy Score:  {acc:.4f}")
    precision = precision_score(y_test, prediction)
    print(f"Precision Score: {precision:.4f}. What percentage of the predicted frauds were frauds?" )
    recall = recall_score(y_test, prediction)
    print(f"Recall Score:    {recall:.4f}. What percentage of the actual frauds were predicted?")
    print(f"Recall Train Score     {clf.score(X_train, y_train):.4f}")


    mean_train_recall_score    = clf.cv_results_["mean_train_recall_score"]
    index = np.argmax( mean_train_recall_score )
    print(f"Recall CV Train Score: {mean_train_recall_score[index]:.4f}" )
    mean_test_recall_score    = clf.cv_results_["mean_test_recall_score"]
    index = np.argmax( mean_test_recall_score )
    print(f"Recall CV Test Score:  {mean_test_recall_score[index]:.4f}" )