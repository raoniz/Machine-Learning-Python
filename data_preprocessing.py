# Data pre-processing

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt  # plot nice charts
import pandas as pd  # import and manage data sets

datasets = pd.read_csv('Data.csv')
X = datasets.iloc[:, :-1].values  #: means select all rows, :-1 means select all but last column ie dont take last column
# X is independent variable matrix of features
y = datasets.iloc[:, 3].values
# y is dependent variable vector matrix

# taking care of missing data
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy="mean", axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])  # this tranforms and replaces the missing values with the mean

# encoding the categorical data
# labelEncoder labels only encodes the value without thinking if there is any order or not
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # preprocessing library

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])  # selecting the country column and all rows selected
# we fitted the labelling code for X object to the first column: country for our matrix X1 and all this
# returns the first column country of the matrix X encoded
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# splitting the dataset into training and test data
from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling 2 methods standardization , normalization
from sklearn.preprocessing import StandardScaler

scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
# train set needs to be fit and transformed
# data set needs only to be transformed
X_test = scale_X.transform(X_test)
