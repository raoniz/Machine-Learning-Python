# Data pre-processing template

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt  # plot nice charts
import pandas as pd  # import and manage data sets

datasets = pd.read_csv('Data.csv')
X = datasets.iloc[:, :-1].values  #: means select all rows, :-1 means select all but last column ie dont take last column
# X is independent variable matrix of features
y = datasets.iloc[:, 3].values
# y is dependent variable vector matrix


# splitting the dataset into training and test data
from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling 2 methods standardization , normalization
#from sklearn.preprocessing import StandardScaler

#scale_X = StandardScaler()
#X_train = scale_X.fit_transform(X_train)
# train set needs to be fit and transformed
# data set needs only to be transformed
#X_test = scale_X.transform(X_test)
# sc_y=StandardScaler()
# y_train=sc_y.fit_transform(y_train)
# train set needs to be fit and transformed
# test set needs only to be transformed

# taking care of missing data and encoding categorical data is excluded
# missing data to be solved ad hoc
# encoding categorical data is rarely there but sometimes if they occur we have to encode them
# feature scaling is commented because generally libraries autometically do it
