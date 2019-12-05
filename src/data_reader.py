from sklearn.preprocessing.data import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
import os

path = os.path.dirname(os.path.realpath(__file__))
file1 = path + "/../data/creditcard_part1.csv"
file2 = path + "/../data/creditcard_part2.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df = pd.concat((df1, df2), ignore_index=True)

print(f"Indices: {df.index}")
print(f"Header: {df.columns.values}\n")

print(df.info(),"\n")

pd.set_option("precision", 3)
print( f" {df.loc[:, ['Time', 'Amount']].describe()}\n" )

### Amount has an average credit card transaction around 88 dollars
### And the biggest transaction of 25691.160 dollarsssssss........

"""
sb.distplot(df["Amount"])
plt.show()


sb.distplot(df["Time"])
plt.show()

exit()
"""
class_counts = df.Class.value_counts()
num_fraudulent = class_counts[1]
num_non_fraudulent = class_counts[0]
print(f"Fraudulent: {num_fraudulent}")
print(f"Non-Fraudulent: {num_non_fraudulent}")
print(f"Ratio: {(num_fraudulent/num_non_fraudulent)*100:.3f}%\n")

plt.bar(class_counts.index, [num_non_fraudulent, num_fraudulent])
plt.xticks(class_counts.index, ('Non-num_fraudulent','Fraudulent'))
plt.show()


sb.heatmap( data=df.corr(), cmap="viridis", annot=False)
plt.show()


### Undersampling
indices = np.arange(0, num_fraudulent)

frauds_df = df.loc[ df["Class"] == 1 ]
non_frauds_df = df.loc[ df["Class"] == 0 ]

indices = np.arange(0, num_non_fraudulent)
indices = np.random.choice(indices, num_fraudulent, replace=False)

#non_frauds_df = non_frauds_df.loc[ indices ]
non_frauds_df = non_frauds_df.reindex( indices )

under_df  = pd.concat((frauds_df, non_frauds_df), ignore_index=True)
df = under_df.sample(frac=1).reset_index(drop=True)




## There are no categories in the dataset, so no need to do one-hot encoding.
X = df.loc[:, df.columns != 'Class'].values
y = df.loc[:, df.columns == 'Class'].values.ravel()


#### StandardScaler is more useful for classification, and Normalizer is more useful for regression.
standard_scaler = StandardScaler()
X = standard_scaler.fit_transform(X)


### Do undersampling to fix imbalanced class



### Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)


### Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score

logreg = LogisticRegression(random_state=4,
                            solver='lbfgs',
                            multi_class='multinomial',
                            max_iter=1000)


from sklearn.model_selection import GridSearchCV
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
>>>>>>> 22ab9fbd6971bb8f0530fe614b1e044c1cb277a6
