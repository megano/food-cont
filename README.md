# Can We Predict Cooking Style?

# Background
I love making food and becoming a better cook by practicing new cooking styles. But when I get home after a long day the last thing I want to do is research recipes. I still need to eat, so to solve the recipe picking problem I made a dinner plan generator. But the planner doesn’t have a “cooking style” feature, so it’s not helping me get any better at cooking. I have 1 million recipes with no cooking style labels, and need a way to label them so my dinner planner can help me build up cooking skills.

# Challenge
Given factors like total time to make recipe, grocery list, and cuisine type can we use machine learning to infer cooking style?

# Planned Approach
Data: Recipe metadata from yummly.com API. Yummly data has 1 million recipes and features like grocery list, recipe name, ID, cuisine, and course (ex: dinner, lunch, salads). Note: data does not contain a label for cooking style or the text description of how to make the dish. I will subset the 1 million recipes, identifying ones that have a word in their name that matches a cooking style to create a labeled data set. Cooking style labels are terms like “blanch” or “roast” from allrecipes.com’s cooking school.

EDA: scraping, mongoDB, python, pandas.
Scraping to get the JSON data from the API.
Python, pandas and mongoDB to store and explore data, do feature engineering.

Model: apply at least 2 supervised learning methods (including linear regression, random forest).

# Model evaluation/metrics
From the labeled data of cooking styles, I’ll split the data into training and testing. Cooking style words in recipe names will be removed from the test data set, and used as ground truth to score model performance on predicted vs actual style. The model will also be tested on a new set of recipes that never had a style word in their name to infer how well it might generalize on the broader set of 1 million recipes.

# Outcome
TBD.
