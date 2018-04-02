# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)  # if dataset does not contain header title
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])
# apriori uses list of list ie 2d array of transactions where
# list containing different transactions(rows containing transactions) each one put in a list(items are filled in the row of the transaction)
# one big list containing all different transactions and each transaction is going to be a list itself

# apyori requires string input format

# Training Apriori on the dataset
# min support = min no of times the product must appear
# min confidence is the no of times the rules will be correct ie if c = 0.8,
# 80% of the time the rule is correct
# lift is the upgrade or improvement of product buying with the help of rule
from apyori import apriori

rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)

# Visualising the results
results = list(rules)
