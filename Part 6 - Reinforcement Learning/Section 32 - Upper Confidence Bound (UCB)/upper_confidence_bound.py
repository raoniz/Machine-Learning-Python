# Upper Confidence Bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
import math

N = 10000  # no of users ads were shown or no of users who were shown the ad
d = 10  # no of ad versions
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0  # index for the ad that is selected for having max upper bound confidence
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):  # ad[i] was selected at least once then we will use this strategy as 
            # for the first 10 rounds there is no strategy and we select 1st ad for round 1, and so upto round 10
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3 / 2 * math.log(n + 1) / numbers_of_selections[i])  # log(n+1) cuz index starts from 0 but in reality it is round 1, and log(0) not defined
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400  # the upper_bound is set as 10^400 so that they get selected at least once
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)  # ad that is selected with max ucb in the round n
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1  # the ad that is selected, no of times selection increases
    reward = dataset.values[n, ad]  # reward of the ad that is selected, it could either 0 or 1 as given in the dataset
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward  # sum of rewards of the ad that is selected upto round n
    total_reward = total_reward + reward  # total reward is updated after each round

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
