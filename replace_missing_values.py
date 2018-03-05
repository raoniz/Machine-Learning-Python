import numpy as np
import matplotlib.pyplot as plt  # plot nice charts
import pandas as pd  # import and manage data sets

datasets = pd.read_csv('Data.csv')
X = datasets.iloc[:,
    :-1].values  #: means select all rows, :-1 means select all but last column ie dont take last column
# X is independent variable matrix of features
y = datasets.iloc[:, 3].values
# y is dependent variable vector matrix

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy="mean", axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])  # this transforms and replaces the missing values with the mean
