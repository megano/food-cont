import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

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
df_roast['sw'] = 'roast'
df_bake['sw'] = 'bake'
df_grill['sw'] = 'grill'
df_fried['sw'] = 'fried'
df_braise['sw'] = 'braise'
df_glaze['sw'] = 'glaze'
df_sautee['sw'] = 'sautee'
df_mash['sw'] = 'mash'
df_steam['sw'] = 'steam'
df_scramble['sw'] = 'scramble'

# Combine style df rows to create a new df including all style words.
df = pd.concat([df_roast, df_bake, df_grill, df_braise, df_fried, df_glaze, df_sautee, df_mash, df_steam, df_scramble], ignore_index=True)
df.sw.unique()
df.head()

# Create a function to count missing values in df:
def num_missing(x):
  return sum(x.isnull())

# Count mising values per column:
print "Missing values per column:"
print df.apply(num_missing, axis=0) #axis=0 defines that function is to be applied on each column

# Count mising values per row:
print "\nMissing values per row:"
print df.apply(num_missing, axis=1).head() #axis=1 defines that function is to be applied on each row

# Drop rows with missing values (ex: cooking time)
df.dropna(axis=0, how='any', inplace=True)
df.info()




# Hold out 25% of data to test using a randomly generated number to select training data
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

df.head()

# Create two new dfs, training and test
train, test = df[df['is_train']==True], df[df['is_train']==False]

# # Check amount of data in training and testing dfs
# print('Number of observations in the training data:', len(train))
# print('Number of observations in the test data:',len(test))

df.columns
'''
Index([u'totalTimeInSeconds', u'ingredients', u'id', u'sw', u'is_train'], dtype='object'
'''

# Convert ingredients to numeric value
# s = df['ingredients']
# # pd.get_dummies(s.apply(pd.Series).stack()).sum(level=0)
# pd.get_dummies(s.apply(pd.Series), prefix='', prefix_sep='').sum(level=0, axis=1)

# # Create feature column list
features = df.columns[0:2]
# Check list to make sure predictor is not included
features

# Train['sw'] contains the cooking style words. Convert each word into a digit.
y = pd.factorize(train['sw'])[0]
# Look for digits 0-9 to check that all 10 style words are present in data set.
y

# Create a random forest classifier.
clf = RandomForestClassifier()

# Train the classifier
clf.fit(train[features], y)
'''
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
            verbose=0, warm_start=False)
'''

# Apply trained classifier to the test data
clf.predict(test[features])

# View the predicted probabilities of the first 10 observations
clf.predict_proba(test[features])[0:10]
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

# Evaluate RF classifier. Compare predicted to actual.
preds = df.sw[clf.predict(test[features])]

# View first 10 predicted classes
preds[0:10]
'''
0    roast
0    roast
2    roast
0    roast
0    roast
0    roast
0    roast
0    roast
0    roast
0    roast
'''

# View first 10 actual classes
test['sw'].head(10)
'''
0     roast
3     roast
6     roast
7     roast
9     roast
10    roast
17    roast
19    roast
23    roast
24    roast
'''

# Create confusion matrix
pd.crosstab(test['sw'], preds, rownames=['Act'], colnames=['Pred'])

# View features and importance scores
list(zip(train[features], clf.feature_importances_))
