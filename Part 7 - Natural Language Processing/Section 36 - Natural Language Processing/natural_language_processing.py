# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Cleaning the texts
import re
import nltk

nltk.download('stopwords')  # list of words that dont add any value like articles, eg this, the
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []  # corpus in nlp is commond word for collection of text, text can be anything: article, html pages etc
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])  # removing characters which are not alphabets and replacing them with ' '
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()  # this library is used to keep the root word eg loved, loving is converted to love
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]  # looping in set is faster
    review = ' '.join(review)  # converting list back to string
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()  # this creates a sparse matrix ie frequent words as column as whether the word exists in the review or not
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
