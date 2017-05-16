import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model


'''
Data: Recipe metadata from yummly.com API.
Yummly data has 1 million recipes and features like grocery list, recipe name,
ID, cuisine, and course (ex: dinner, lunch, salads). Note: data does not
contain a label for cooking style or the text description of how to make the dish.

Plan:
Subset the 1 million recipes, identify ones that have a word in their
name that matches a cooking style to create a labeled data set. Cooking style
labels are terms like “blanch” or “roast” from allrecipes.com’s cooking school.

EDA:
scraping, mongoDB, python, pandas. Scraping to get the JSON data from the API.
Python, pandas and mongoDB to store and explore data, do feature engineering.

Model: apply at least 2 supervised learning methods
(including linear regression, random forest).

'''

# Load the dataset - json
# with open('style-all.json') as data_file:
# data = json.load(data_file)

# Load the data set - csv
style = pd.read_csv('style-all.csv')

# Use only one feature
style_X = style.data[:, np.newaxis, 2]

# Split the data into training/testing sets
style_X_train = style_X[:-20]
style_X_test = style_X[-20:]

# Split the targets into training/testing sets
style_y_train = style.target[:-20]
style_y_test = style.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(style_X_train, style_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(style_X_test) - style_y_test) ** 2))
# variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(style_X_test, style_y_test))

# Plot outputs
plt.scatter(style_X_test, style_y_test,  color='black')
plt.plot(style_X_test, regr.predict(style_X_test), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
