import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import numpy as np
import random

'''
Data: Recipe metadata from yummly.com API.
Yummly data has 1 million recipes and features like grocery list, recipe name,
ID, cuisine, and course (ex: dinner, lunch, salads). Note: data does not
contain a label for cooking style or the text description of how to make the dish.

Plan:
Subset the 1 million recipes, identify ones that have a word in their
name that matches a cooking style to create a labeled data set. Cooking style
labels are terms like blanch or roast from allrecipes.com cooking school.

EDA:
scraping, mongoDB, python, pandas. Scraping to get the JSON data from the API.
Python, pandas and mongoDB to store and explore data, do feature engineering.

Model: apply at least 2 supervised learning methods
(including logistic regression, random forest - make sure to use classification kind).
Show ROC curve for the 2 similar styles and 2 different styles.
'''

# Load the dataset - json
# with open('style-all.json') as data_file:
# data = json.load(data_file)

# Load the data set - csv
# skip importing the unnecessary columns for: id, sourceDisplayName, rating.
# TODO: more_cols_to_keep = 'flavors', 'totalTimeInSeconds', 'recipeName', 'attributes'
df_roast = pd.read_csv('../data/style-roast.csv', header='infer', usecols=['id', 'totalTimeInSeconds'], verbose=True)
df_bake = pd.read_csv('../data/style-bake.csv', header='infer', usecols=['id', 'totalTimeInSeconds'], verbose=True)

# Add new column for style word
df_roast['sw'] = 'roasted'
df_bake['sw'] = 'baked'

# Append roasted df with baked dfs rows with flag to avoid duplicating index values
df = df_roast.append(df_bake, ignore_index=True)
df.head()
df.reset_index()

# drop column with recipe name and number
df.drop("id", axis=1, inplace=True)

# Convert string categorical names to integers
integerized_data = preprocessing.LabelEncoder().fit_transform(df['sw'])

# View the integerized data array
integerized_data

# Convert integer categorical representations to OneHot encodings
preprocessing.OneHotEncoder().fit_transform(integerized_data.reshape(-1,1)).toarray()

# Convert cook time from seconds to hours
df['totalTimeInHours'] = df['totalTimeInSeconds']/3600

'''
df.isnull().sum()
# 84 of 8132 have null values.

# Let's take a look at cooking time distribution
df.info()
8216 entries, 8132 non-null.

df.describe()
count         8132.000000       8132.000000
mean          3202.865224          0.889685
std           5297.563288          1.471545
min            120.000000          0.033333
25%           1800.000000          0.500000
50%           2700.000000          0.750000
75%           3600.000000          1.000000
max         172800.000000         48.000000

df[df['totalTimeInHours'] > 1].count()
1677 of 8132 roasted and baked recipes are greater than 1 hr cook time.
'''

# Drop the 84 entries with no cook time value
df.dropna(axis=0, how='any', inplace=True)

# Create our predictor (independent) variable
# and our response (dependent) variable
X = df['totalTimeInHours'] # can add in other variables
y = df['sw']

# Split predictor and response into training/testing sets
## Use stratified sampling to get same representation in test and train sets
sss = StratifiedShuffleSplit(n_splits=2, test_size=0.5, random_state=0)
sss.get_n_splits(X, y)
print(sss)

for train_index, test_index in sss.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# # Create a scaler object to standardize features
# sc = StandardScaler()
#
# # Fit the scaler to the training data and transform
# X_train_std = sc.fit_transform(X_train)
#
# # Apply the scaler to the test data
# X_test_std = sc.transform(X_test)

# Run logistic regression with L1 penalty with various regularization strengths
C = [10, 1, .1, .001]

for c in C:
    clf = LogisticRegression(penalty='l1', C=c)
    clf.fit(X_train, y_train)
    print('C:', c)
    print('Coefficient of each feature:', clf.coef_)
    print('Training accuracy:', clf.score(X_train, y_train))
    print('Test accuracy:', clf.score(X_test, y_test))
    print('')
