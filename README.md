# machine_learning_3
Machine learning project in FYS-STK3155/4155 at the University of Oslo. We also made a report for this project where you can read a detailed description of the methods and our results. 

[FYS-STK3155/4155 Project 3 Report](https://github.com/michaesb/machine_learning_3/)


# Description

## Folder structure

**data** - where we find the credit card info

**plots** - collection of the pictures used for the rapport

**src** - where we find the methods, test and data extraction.

### Standalone files

[Project3.pdf](https://github.com/michaesb/machine_learning_3/blob/master/Project3.pdf) - the task given in subject FYS-STK3155/4155 at the University of Oslo

[FYS-STK_Project3_Report.pdf](ttps://github.com/michaesb/machine_learning_3/blob/master/FYS-STK_Project3_Report.pdf) - Our report for the project

## Data
We look at detection of fraudulent transactions for credit card data. We extract data from the data folder, which contains 2 files. (The data file was split in two parts in order to circumvent GitHub's file size limit).
The dataset contain the data from credit card transactions over 48 hours in Europe. These contain a number of parameters and if the transaction was fraudulent or non-fraudulent. Fraudelent is classified as a 1 and non-fraudulent is a 0. The ratio between fraudulent and non-fraudulent is 0.173 %.

Due to privacy concerns, we can only identify a few features, e.g. time and amount. The unknown parameters are classified as V1, V2, ..., V28. According to the Kaggle description, the dataset features has already been simplified with a PCA (principal component analysis) reduction.

The dataset was acquired from the site
[Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)

We process the data in the data_process.py file in the src-folder. In order to view the data processing visualized, you can run the file [src/data_visualizer.py](https://github.com/michaesb/machine_learning_3/blob/master/src/data_visualizer.py).


## Methods

We have here performed four machine learning methods in order to make a model of the transactions.
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

We have made a test function for the file handling the data processing, but since we used Scikit-Learns packages which has been heavily tested, we did not create tests for these.
In order to run the test execute the command below
```
pytest -v
```

## Dependencies
This repository uses Python 3, and the following packages needs to be installed:
* NumPy
* MatPlotLib
* Seaborn
* Scikit-Learn
* Pandas
* Graphviz
```
pip install numpy matplotlib seaborn sklearn pandas graphviz
```

### Authors

* **Michael Bitney**
* **Magnus Grøndalen**
