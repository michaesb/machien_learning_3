# machine_learning_3
Machine learning project in FYS-STK4155/3155 at University of Oslo. We have made a rapport of this
# Description

### Folder structure

data - where we find the credit card info

plots - collection of the pictures used for the rapport

src  - where we find the methods, test and data extraction.

### Singular files

Project3.pdf - the task given by University of Oslo

_____________ - Our rapport on the methods result


## Data
Here we look at detection of fraud on credit card data. We extract data from the data folder, which contains 2 files. (The files was split in  two in order to be able to upload the data to Github).
The files together contain the data from credit card transactions over 24 hours in Europe. These contain a number of parameters and if the transaction was fraudulent or non-fraudulent. Fraudelent is classified as a 1 and non-fraudulent is a 0. The ratio between fraudulent and non-fraudulent is 0.173 %.

Due to privacy concerns, we can only identify a few features E.g. time and amount. The Unknown parameters are classified as V1, V2, ..., V28. According to the Kaggle description, the dataset has already been simplified with a PCA (principal component analysis).

The dataset was acquired from the site
[Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)

We process the data in the data_process.py file in the src-folder. In order to view the data processing visualized, you can check out ...


## Methods

We have here performed 4 methods in order to make a model of the transactions.
The methods below have investigated by plotting and looking at mutliple input parameters, but for some like neural network there are huge amount of variations in these, so we have used scikitlearns gridsearch to test out these parameters.

### Logistic Regression


```
python src/logistic_regression.py
```


### Neural Network

```
python src/neural_network.py
```

### Decision tree

```
python src/decision_tree.py
```

### Random Forest

```
python src/random_forest.py
```


## Test

We have made a testfunction for the data processor, but since we used scikitlearns packages which has been heavily tested, we have not created test for these.
In order to run the test execute the command below
```
pytest -v
```




### Authors

* **Michael Bitney**
* **Magnus Grøndalen**
