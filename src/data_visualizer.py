#!/usr/bin/env python
from sklearn.preprocessing.data import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
import os

"""
A program prints and plots different properties of the datasets and process it

"""


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
plt.title("Distribution of Amount", size=16)
plt.xlabel("Amount", size=14)
plt.show()


sb.distplot(df["Time"])
plt.title("Distribution of Time", size=16)
plt.xlabel("Time (Seconds)", size=14)
plt.show()
"""


class_counts = df.Class.value_counts()
num_fraudulent = class_counts[1]
num_non_fraudulent = class_counts[0]
print(f"Fraudulent: {num_fraudulent}")
print(f"Non-Fraudulent: {num_non_fraudulent}")
print(f"Ratio: {(num_fraudulent/num_non_fraudulent)*100:.3f}%\n")

plt.bar(class_counts.index, [num_non_fraudulent, num_fraudulent])
plt.xticks(class_counts.index, ('Non-fraudulent','Fraudulent'))
plt.title("Non-Fraudulent & Fraudulent transactions total", size=16)
plt.ylabel("Amount", size=14)
plt.show()


sb.heatmap( data=df.corr(), cmap="viridis", annot=False)
plt.show()


### Undersampling
# indices = np.arange(0, num_fraudulent)

# frauds_df = df.loc[ df["Class"] == 1 ]
# non_frauds_df = df.loc[ df["Class"] == 0 ]

# indices = np.arange(0, num_non_fraudulent)
# indices = np.random.choice(indices, num_fraudulent, replace=False)

# #non_frauds_df = non_frauds_df.loc[ indices ]
# non_frauds_df = non_frauds_df.reindex( indices )

# under_df  = pd.concat((frauds_df, non_frauds_df), ignore_index=True)
# df = under_df.sample(frac=1).reset_index(drop=True)

####


## There are no categories in the dataset, so no need to do one-hot encoding.
X = df.loc[:, df.columns != 'Class'].values
y = df.loc[:, df.columns == 'Class'].values.ravel()


#### StandardScaler is more useful for classification, and Normalizer is more useful for regression.
standard_scaler = StandardScaler()
X = standard_scaler.fit_transform(X)

indices_nonfraud = np.where(y==0)[0]
indices_fraud = np.where(y==1)[0]

np.random.shuffle(indices_nonfraud)
indices_nonfraud_under = indices_nonfraud[:num_fraudulent]
indices_under = np.concatenate( (indices_fraud, indices_nonfraud_under) )
np.random.shuffle(indices_under)


X = X[indices_under]
y = y[indices_under]

plt.bar([0,1], [len(indices_nonfraud_under), len(indices_fraud)])
plt.xticks(class_counts.index, ('Non-fraudulent','Fraudulent'))
plt.title("Non-Fraudulent & Fraudulent transactions total", size=16)
plt.ylabel("Amount", size=14)
plt.show()

sb.distplot(X[:,-1])
plt.title("Distribution of Amount", size=16)
plt.xlabel("Amount", size=14)
plt.show()

sb.distplot(X[:,0])
plt.title("Distribution of Time", size=16)
plt.xlabel("Time (Seconds)", size=14)
plt.show()

import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(y_test, predictions, normalize=True)
plt.show()
### Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)
