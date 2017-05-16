import pandas as pd
import numpy as np

'''
Data: Recipe metadata from yummly.com API.
Yummly data has 1 million recipes and features like grocery list, recipe name,
ID, cuisine, and course (ex: dinner, lunch, salads). Note: data does not
contain a label for cooking style or the text description of how to make the dish.

Plan:
Subset the 1 million recipes, identifying ones that have a word in their
name that matches a cooking style to create a labeled data set. Cooking style
labels are terms like “blanch” or “roast” from allrecipes.com’s cooking school.

EDA:
scraping, mongoDB, python, pandas. Scraping to get the JSON data from the API.
Python, pandas and mongoDB to store and explore data, do feature engineering.

Model: apply at least 2 supervised learning methods
(including linear regression, random forest).

TODO:
- copy source data from other folder
- make cooking words list
- set up mongodb
- set up mongo query pipelines
-- total # recipes
-- # recipes that have 1 or more cooking words
-- # recipes that have no cooking words
-- distribution of recipes per cooking word (need to remove any words?)
-- distribution of cooking words in recipies
'''

RECIPE_CSV =    'all_recipes_unfiltered.csv'
RECIPE_CSV_COL_NAMES = ['id', 'name', 'course', 'ingredient']
