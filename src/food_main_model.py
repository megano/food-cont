import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import re

usecols=['id', 'totalTimeInSeconds', 'ingredients']

# Import style word csv files
df_roast = pd.read_csv('../data/style-roast.csv', header='infer', usecols=usecols, index_col="id")
df_bake = pd.read_csv('../data/style-bake.csv', header='infer', usecols=usecols, index_col="id")
df_grill = pd.read_csv('../data/style-grill.csv', header='infer', usecols=usecols, index_col="id")
df_fried = pd.read_csv('../data/style-fried.csv', header='infer', usecols=usecols, index_col="id")
df_braise = pd.read_csv('../data/style-braise.csv', header='infer', usecols=usecols, index_col="id")
df_glaze = pd.read_csv('../data/style-glaze.csv', header='infer', usecols=usecols, index_col="id")
df_sautee = pd.read_csv('../data/style-sautee.csv', header='infer', usecols=usecols, index_col="id")
df_mash = pd.read_csv('../data/style-mash.csv', header='infer', usecols=usecols, index_col="id")
df_steam = pd.read_csv('../data/style-steam.csv', header='infer', usecols=usecols, index_col="id")
df_scramble = pd.read_csv('../data/style-scramble.csv', header='infer', usecols=usecols, index_col="id")

# To each df, add new column with style word, which we are going to try to predict
df_roast['style_word'] = 'roast'
df_bake['style_word'] = 'bake'
df_grill['style_word'] = 'grill'
df_fried['style_word'] = 'fried'
df_braise['style_word'] = 'braise'
df_glaze['style_word'] = 'glaze'
df_sautee['style_word'] = 'sautee'
df_mash['style_word'] = 'mash'
df_steam['style_word'] = 'steam'
df_scramble['style_word'] = 'scramble'

# Combine style df rows to create a new df including all style words.
df = pd.concat([df_roast, df_bake, df_grill, df_steam, df_braise, df_fried, df_glaze, df_sautee, df_mash, df_scramble], ignore_index=True)
df.style_word.unique()
df.head()

# Create a variable for flour being in ingredient list
flour = [bool(re.search('.*flour.*',x)) for x in  df['ingredients']]
# Engineer new column with True and False values for flour present in ingredient list
df['has_flour'] = flour

# Change data type of df column to 0 and 1
df.has_flour = df.has_flour.astype(np.float64)

# Add flour column onto main df
pd.concat([df, df['has_flour']], axis=1)
# Drop ingredients column
df.drop('ingredients', axis=1, inplace=True)

# # Create a function to count missing values in df:
# def num_missing(x):
#   return sum(x.isnull())

# # Count missing values per column:
# print "Missing values per column:"
# print df.apply(num_missing, axis=0) #axis=0 defines that function is to be applied on each column
#
# # Count mising values per row:
# print "\nMissing values per row:"
# print df.apply(num_missing, axis=1).head() #axis=1 defines that function is to be applied on each row

# Drop rows with missing values (ex: cooking time)
df.dropna(axis=0, how='any', inplace=True)
df.info()

# Convert cook time from seconds to hours
df['totalTimeInHours'] = df['totalTimeInSeconds']/3600
# Drop cook time in seconds column
df.drop('totalTimeInSeconds', axis=1, inplace=True)

# Plot cook time in hours for each cooking style
df.boxplot(column="totalTimeInHours",by="style_word")
plt.show()

# # Re-count mising values per column to make sure there's no null values anymore
# print "Missing values per column:"
# print df.apply(num_missing, axis=0) #axis=0 defines that function is to be applied on each column

df.columns

# Create our predictor (independent) variable
X = df[[col for col in df.columns if col != 'style_word']]
# And our response (dependent) variable
y = df['style_word']

# Split training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

# Run logistic regression with L1 penalty (aka least absolute deviations)
# with various regularization strengths
C = [10, 1, .1, .001]
for c in C:
    clf = LogisticRegression(penalty='l1', C=c)
    clf.fit(X_train, y_train)
    print('C:', c)
    print('Coefficient of each feature:', clf.coef_)
    print('Training accuracy:', clf.score(X_train, y_train))
    print('Test accuracy:', clf.score(X_test, y_test))
    print('')

# Create a random forest classifier.
clf = RandomForestClassifier()

# Train the classifier
clf.fit(X_train, y_train)

clf.score(X_train, y_train)
clf.score(X_test, y_test)

y_test_predicted = clf.predict(X_test)

# Create confusion matrix
print pd.crosstab(y_test, y_test_predicted, rownames=['Actual Style'], colnames=['Predicted Style'])

# Apply trained classifier to the test data
clf.predict(X_test)

# View the predicted probabilities of the first 10 observations
clf.predict_proba(X_test)[0:10]

# Make crosstab to check hypothesis that cooking time affects cooking style
pd.crosstab(df['totalTimeInHours'], df['style_word'],margins=True)
def percConvert(ser):
    return ser/float(ser[-1])
    pd.crosstab(df['totalTimeInHours'], df['style_word'],margins=True).apply(percConvert, axis=1)

# View X and importance scores
print list(zip(X_train, clf.feature_importances_))
