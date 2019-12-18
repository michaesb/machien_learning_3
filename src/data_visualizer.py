#!/usr/bin/env python

## Importing needed packages
from sklearn.preprocessing.data import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
import os

"""
A program prints and plots different properties of the datasets and process it

"""

## Getting and reading csv-data files into a pandas dataframe
path = os.path.dirname(os.path.realpath(__file__))
file1 = path + "/../data/creditcard_part1.csv"
file2 = path + "/../data/creditcard_part2.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df = pd.concat((df1, df2), ignore_index=True)

## Printing the indices and features
print(f"Indices: {df.index}")
print(f"Header: {df.columns.values}\n")

## Printing info about the dataset
print(df.info(),"\n")

## Looking at the unscaled features Time and Amount
pd.set_option("precision", 3)
print( f" {df.loc[:, ['Time', 'Amount']].describe()}\n" )

## Distribution plot of the Amount data
sb.distplot(df["Amount"])
plt.title("Distribution of Amount", size=16)
plt.xlabel("Amount", size=14)
plt.show()

## Distribution plot of the Time data
sb.distplot(df["Time"]/(60*60))
plt.title("Distribution of Time", size=16)
plt.xlabel("Time (Hrs)", size=14)
plt.show()

## Looking at the class balances
class_counts = df.Class.value_counts()
num_fraudulent = class_counts[1]
num_non_fraudulent = class_counts[0]
print(f"Fraudulent: {num_fraudulent}")
print(f"Non-Fraudulent: {num_non_fraudulent}")
print(f"Ratio: {(num_fraudulent/num_non_fraudulent)*100:.3f}%\n")

## Plotting the class imbalance
plt.bar(class_counts.index, [num_non_fraudulent, num_fraudulent])
plt.xticks(class_counts.index, ('Non-fraudulent','Fraudulent'))
plt.title("Non-Fraudulent & Fraudulent transactions total", size=16)
plt.ylabel("Amount", size=14)
plt.show()

## Plotting the correlation matrix. (Dataset is already PCA'd)
sb.heatmap( data=df.corr(), cmap="viridis", annot=False)
plt.show()

## There are no categories in the dataset, so no need to do one-hot encoding.

## Splitting the dataset into design matrix X and targets y
X = df.loc[:, df.columns != 'Class'].values
y = df.loc[:, df.columns == 'Class'].values.ravel()


## Scaling the data (Most for Time and Amount)
standard_scaler = StandardScaler()
X = standard_scaler.fit_transform(X)


## Randomized undersampling method
indices_nonfraud = np.where(y==0)[0]
indices_fraud = np.where(y==1)[0]
np.random.shuffle(indices_nonfraud)
indices_nonfraud_under = indices_nonfraud[:num_fraudulent]
indices_under = np.concatenate( (indices_fraud, indices_nonfraud_under) )
np.random.shuffle(indices_under)

## Using indices from undersampling method to create new balanced dataset
X_under = X[indices_under]
y_under = y[indices_under]

## Looking at the class balance again, now for undersampled data
plt.bar([0,1], [len(indices_nonfraud_under), len(indices_fraud)])
plt.xticks(class_counts.index, ('Non-fraudulent','Fraudulent'))
plt.title("Non-Fraudulent & Fraudulent transactions total", size=16)
plt.ylabel("Amount", size=14)
plt.show()

## New distrubution plots for undersampled data
sb.distplot(X_under[:,-1])
plt.title("Distribution of Amount", size=16)
plt.xlabel("Amount", size=14)
plt.show()

sb.distplot(X_under[:,0])
plt.title("Distribution of Time", size=16)
plt.xlabel("Time (Seconds)", size=14)
plt.show()
