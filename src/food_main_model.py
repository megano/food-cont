import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
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

# Vectorize
countvec = CountVectorizer()
countvec.fit_transform(df.ingredients)
df_ing = pd.DataFrame(countvec.fit_transform(df.ingredients).toarray(), columns=countvec.get_feature_names())
# print df_ing
df_ing2 = pd.concat([df, df_ing], ignore_index=True)
#
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

# # Count mising values per column:
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
# df.boxplot(column="totalTimeInHours",by="style_word")
# plt.show()

# # Re-count mising values per column to make sure there's no null values anymore
# print "Missing values per column:"
# print df.apply(num_missing, axis=0) #axis=0 defines that function is to be applied on each column

df.columns
'''
Index([u'style_word', u'has_flour', u'totalTimeInHours', u'is_train'], dtype='object')
'''

# Create our predictor (independent) variable
X = df[[col for col in df.columns if col != 'style_word']]
# And our response (dependent) variable
y = df['style_word']

# Split training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

# Create a random forest classifier.
clf = RandomForestClassifier()

# Train the classifier
clf.fit(X_train, y_train)

clf.score(X_train, y_train)
clf.score(X_test, y_test)

y_test_predicted = clf.predict(X_test)

# Create confusion matrix
pd.crosstab(y_test, y_test_predicted, rownames=['Actual Style'], colnames=['Predicted Style'])

'''

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
            verbose=0, warm_start=False)
'''

# Apply trained classifier to the test data
clf.predict(X_test)

# View the predicted probabilities of the first 10 observations
clf.predict_proba(X_test)[0:10]
'''
array([[ 0.36759677,  0.2717327 ,  0.23052817,  0.01589377,  0.05490892,
         0.0175377 ,  0.01683401,  0.01362793,  0.01134004,  0.        ],
       [ 0.42672385,  0.32509248,  0.16696949,  0.02366823,  0.02549009,
         0.00737477,  0.00156891,  0.00976315,  0.01334903,  0.        ],
       [ 0.27253369,  0.24157831,  0.30210094,  0.01410009,  0.06548224,
         0.02583002,  0.02466769,  0.02418895,  0.01752625,  0.01199181],
       [ 0.41551259,  0.34720272,  0.1541295 ,  0.03622082,  0.01457573,
         0.01468154,  0.00203982,  0.01266005,  0.00297723,  0.        ],
       [ 0.32637624,  0.30550716,  0.23181637,  0.01269785,  0.05041839,
         0.00986817,  0.01495267,  0.02357919,  0.02216499,  0.00261899],
       [ 0.32637624,  0.30550716,  0.23181637,  0.01269785,  0.05041839,
         0.00986817,  0.01495267,  0.02357919,  0.02216499,  0.00261899],
       [ 0.4820574 ,  0.32136301,  0.13861515,  0.01514702,  0.01770235,
         0.01558466,  0.        ,  0.00496306,  0.00456735,  0.        ],
       [ 0.44203248,  0.32612639,  0.14684682,  0.0085581 ,  0.036069  ,
         0.02018333,  0.00441845,  0.00841307,  0.00735235,  0.        ],
       [ 0.39371948,  0.32941   ,  0.17573998,  0.01874995,  0.02881364,
         0.01139362,  0.00758651,  0.01972719,  0.01014132,  0.00471831],
       [ 0.42672385,  0.32509248,  0.16696949,  0.02366823,  0.02549009,
         0.00737477,  0.00156891,  0.00976315,  0.01334903,  0.        ]])
'''


# Make crosstab to check hypothesis that cooking time affects cooking style
# pd.crosstab(df['totalTimeInSeconds'], df['style_word'],margins=True)
def percConvert(ser):
  return ser/float(ser[-1])
  pd.crosstab(df['totalTimeInHours'], df['style_word'],margins=True)\
  .apply(percConvert, axis=1)

# View X and importance scores
print list(zip(X_train, clf.feature_importances_))


# # Run logistic regression with L1 penalty with various regularization strengths
# C = [10, 1, .1, .001]
# for c in C:
#     clf = LogisticRegression(penalty='l1', C=c)
#     clf.fit(X_train, y_train)
#     print('C:', c)
#     print('Coefficient of each feature:', clf.coef_)
#     print('Training accuracy:', clf.score(X_train, y_train))
#     print('Test accuracy:', clf.score(X_test, y_test))
#     print('')
