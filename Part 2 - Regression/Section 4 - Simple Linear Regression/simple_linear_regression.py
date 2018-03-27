# Simple linear regression

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datasets = pd.read_csv('Salary_Data.csv')
X = datasets.iloc[:, :-1].values
# X is independent variable matrix of features
y = datasets.iloc[:, 1].values
# y is dependent variable vector matrix


# splitting the dataset into training and test data
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)

# feature scaling 2 methods standardization , normalization
# from sklearn.preprocessing import StandardScaler
# scale_X = StandardScaler()
# X_train = scale_X.fit_transform(X_train)
# X_test = scale_X.transform(X_test)

# fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting the test set results
y_pred = regressor.predict(X_test)  # this is a vector of predictions of dependent variable

# visualising the test results
plt.scatter(X_train, y_train, color='red')  # og values
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (train)')
plt.xlabel('Years of exp')
plt.ylabel('Salary')
plt.show()

# visualising the test results
plt.scatter(X_test, y_test, color='red')  # og values
plt.plot(X_train, regressor.predict(X_train), color='blue')  # even replaced by xtest not matter as it will generate the same equation
plt.title('Salary vs Experience (test)')
plt.xlabel('Years of exp')
plt.ylabel('Salary')
plt.show()
