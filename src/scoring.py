from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sb
import numpy as np



def scores(prediction, y_test, X_train, y_train, grid_search=None):
    if grid_search is not None:
        print("Best params for Recall Score", grid_search.best_params_)

    acc = accuracy_score(y_test, prediction)
    print(f"Accuracy Test Score:   {acc:.4f}")
    precision = precision_score(y_test, prediction)
    print(f"Precision Test Score:  {precision:.4f}. What percentage of the predicted frauds were frauds?" )
    recall = recall_score(y_test, prediction)
    print(f"Recall Test Score:     {recall:.4f}. What percentage of the actual frauds were predicted?")
    if grid_search is not None:
        print(f"Recall Train Score     {grid_search.score(X_train, y_train):.4f}")

        mean_train_recall_score    = grid_search.cv_results_["mean_train_recall_score"]
        index = np.argmax( mean_train_recall_score )
        print(f"Recall CV Train Score: {mean_train_recall_score[index]:.4f}" )
        mean_test_recall_score    = grid_search.cv_results_["mean_test_recall_score"]
        index = np.argmax( mean_test_recall_score )
        print(f"Recall CV Test Score:  {mean_test_recall_score[index]:.4f}" )

    ax= plt.subplot()
    cm = confusion_matrix(y_test, prediction)
    sb.heatmap(cm, annot=True, ax = ax, fmt="g", cmap="Greens")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    ax.xaxis.set_ticklabels(["Non-fraud", "Fraud"])
    ax.yaxis.set_ticklabels(["Non-fraud", "Fraud"])
    plt.show()
