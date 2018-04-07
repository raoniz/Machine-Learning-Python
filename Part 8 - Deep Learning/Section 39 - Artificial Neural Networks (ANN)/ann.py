# Artificial Neural Network

# Installing Theano     #numerical computation lib, efficient for fast numerical computations, can run on gpu
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# categorical data is converted into dummy variable
# in regression model we will take either ny or cali as 0,1 and when only 1 dummy variable taken into eqn
# as the other is assumed as default when the dmummy var is 0, and the coeff is adjusted in the const
# if 100 variable in a particaular dummy set then only select 99 of them, several independent variables predict each 
# other = multicollinearity
# b0 + ... + b4 * D1, b5 * D2 not included; b0,b4*D1,b5*D2 should not be together
# if there are 2 dummy sets then the rule of removing 1 dummy variable is applied to each set


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])  # country
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])  # gender

# not creating dummy variables for gender as it contains only 2 values and 1 dummy variable will be removed
# which is same as keeping the gender column

#

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]  # removing 1 dummy variable

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# dense function will randmoly initialize the weights close to 0 but !=0
# no of nodes in input layer = no of independent variables
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))

# first hidden layer requires input_dim and the output_dim specifies the number of nodes in the hidden layer
# no of nodes in the hidden layer is generally average of no of input nodes and no of output nodes

# Adding the second hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

# second hidden is not really required for this dataset but for explanation of deep learning we are using this 
# contrived layer
# this does not require input_dim as the first layer has output_dim as 6
# the average node rule is applied to this layer also

# generally hidden layers rectifier functions and output layer sigmoid function(generally for classification or
#  probability of the event eg will he buy the car or not)

# Adding the output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
# if output categorical var of 3 output_dim=33, activation = softmax

# Compiling the ANN
# stochastic gradient descent applied = compiling
# adam type of stochastic gradient descent

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# optimizer is the algo you want to use to find the optimal set of weight
# loss function corresponds to loss function within the stochastic gradient descent, ie Adam 
# in stochastic gradient descent algo, you have to optimize the loss function in order to gain the optimal set of weights
# eg loss function in regression linear model was sum of sqaured difference between y_pred and y_actual, sum of squared errors
# if binary outcome of your dependent variable then dont use binary_crsonnentropy
# if outcome 3 categories then use logarithmic loss function categorical_crossentropy 
# metrics: criterion that you use to evaluate your model
# when the weights are updated after each batch the accuracy criteriaon is used to improve model performance


# Fitting the ANN to the Training set

classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)

# batch_size is the no of rows or transactions that the ANN will store in memory and calculate the 
# cost function for all rows at once ie the no of observations afther which you want to update the weights
# epoch is basically a round when the whole training set is passed through the ANN
# in reality training ANN consists appying the steps 1-6 of the algo over many epochs

# no set rule of thumb for batch_size and nb_epoch

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)  # convert probabilities to true or false

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
