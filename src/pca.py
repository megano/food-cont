import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
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

# Vectorize ingredients
countvec = CountVectorizer()
countvec.fit_transform(df.ingredients)
df_ing = pd.DataFrame(countvec.fit_transform(df.ingredients).toarray(), columns=countvec.get_feature_names())
# Merge ingredient vectors back onto df
df = pd.concat([df, df_ing], axis=1)

# Drop ingredients column
df.drop('ingredients', axis=1, inplace=True)

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

# Scale cook time from 0 to 1 to match other columns
# scaler = MinMaxScaler()
# df['totalTimeInHours']=scaler.fit_transform(df['totalTimeInHours'])
# print df
#
# Create our predictor (independent) variable
not_style = [col for col in df.columns if col != 'style_word']
X = df[not_style]
# And our response (dependent) variable
y = df['style_word']

# View max values for other columns
# X.max().sort_values(ascending=False)

# Scale the data
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=not_style)

# PCA
# Create a pca object with the 2 components as a parameter
pca = decomposition.PCA(n_components=10)
# Fit PCA
pca.fit(X)
# and transform the data
X_pca = pca.transform(X)

# After PCA the data is reduced to 10 features.
# View the shape of new feature data
print X_pca.shape
# View new feature data
print X_pca
