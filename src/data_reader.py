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
df  = pd.concat((df1, df2), ignore_index=True)


print(f"Indices: {df.index}")
print(f"Header: {df.columns.values}\n")

print(df.info(),"\n")

pd.set_option("precision", 3)
print( f" {df.loc[:, ['Time', 'Amount']].describe()}\n" )

### Amount has an average credit card transaction around 88 dollars 
### And the biggest transaction of 25691.160 dollarsssssss........

sb.distplot(df["Time"])
plt.show()

sb.distplot(df["Amount"])
plt.show()

class_counts = df.Class.value_counts()
fraudulent = class_counts[1]
non_fraudulent = class_counts[0]
print(f"Fraudulent: {fraudulent}")
print(f"Non-Fraudulent: {non_fraudulent}")
print(f"Ratio: {(fraudulent/non_fraudulent)*100:.3f}%")

plt.bar(class_counts.index, [non_fraudulent, fraudulent])
plt.xticks(class_counts.index, ('Non-fraudulent','Fraudulent'))
plt.show()


sb.heatmap( data=df.corr(), cmap="viridis", annot=False)
plt.show()



## There are no categories in the dataset, so no need to do one-hot encoding.
X = df.loc[:, df.columns != 'Class'].values
y = df.loc[:, df.columns == 'Class'].values.ravel()


#### StandardScaler is more useful for classification, and Normalizer is more useful for regression.
standard_scaler = StandardScaler()
X = standard_scaler.fit_transform(X)


### Do undersampling to fix imbalanced class