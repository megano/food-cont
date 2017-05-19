# Can We Predict Cooking Style?

# Background
I love making food and becoming a better cook by practicing new cooking styles. But when I get home after a long day the last thing I want to do is research recipes. I still need to eat, so to solve the recipe picking problem I made a dinner plan generator. But the planner doesn’t have a “cooking style” feature, so it’s not helping me get any better at cooking. I have 1 million recipes with no cooking style labels, and need a way to label them so my dinner planner can help me build up cooking skills.

# Challenge
Given factors like total time to make recipe, grocery list, and cuisine type can we use machine learning to infer cooking style?

# Planned Approach
Data: Recipe metadata from yummly.com API. Yummly data has 1 million recipes and features like grocery list, recipe name, ID, cuisine, and course (ex: dinner, lunch, salads). Note: data does not contain a label for cooking style or the text description of how to make the dish. I will subset the 1 million recipes, identifying ones that have a word in their name that matches a cooking style to create a labeled data set. Cooking style labels are terms like “blanch” or “roast” from allrecipes.com’s cooking school.
EDA: scraping, mongoDB, python, pandas. Scraping to get the JSON data from the API. Python, pandas and mongoDB to store and explore data, do feature engineering.
Model: apply at least 2 supervised learning methods (including linear regression, random forest).

# Model evaluation/metrics
From the labeled data of cooking styles, I split the data into training and testing. Cooking style words in recipe names will be removed from the test data set, and used as ground truth to score model performance on predicted vs actual style.

Examples:
No style word: "recipeName" : "20 Minute Honey Garlic Shrimp"
Has style word: "recipeName" : "Baked Apple Pie Roll Ups"

# Outcome
Below are predicted and actual cooking styles. Overall, prediction accuracy was  not bad for certain cooking styles, and way off on others. The model did correctly classify sautee, scramble, steam, bake, grill and roast the majority of the time.

Using the features cooking time and presence of flour in ingredients, the model had a hard time differentiating between styles like bake, grill and roast. This makes sense given that baking and roasting both involve ovens and the main differentiator is whether the ingredients are liquid (ex: cake) or solid (ex: ham).  

Predicted Style  bake  braise  fried  glaze  grill  mash  roast
Actual Style                                                              
bake              681       0     11      3     84     0    112        
braise             18      22      2      0      7     0     24         
fried              48       1     77      0     22     0     18          
glaze              23       0      0     32      5     0      4          
grill              89       1      6      1    562     1    155         
mash               12       0      2      0      1    23     18         
roast             102       0      5      2    121     2    906          
sautee              5       0      1      0     22     0     17         
scramble            9       0      0      0      1     0      2          
steam              14       0      4      0      8     0      8         

Predicted Style  sautee scramble  steam  
Actual Style                      
bake              0     		0      0  
braise            0     		0      0  
fried             0     		0      0  
glaze             0     		0      0  
grill             2     		0      0  
mash              0     		0      0  
roast             1     		1      2  
sautee            19    		0      0  
scramble          1    		14     0  
steam             0     		0     14

Predicted Style  bake  braise  fried  glaze  grill  mash  none  roast
Actual Style                                                                    
bake              404       0     20      6    143     0     2    312       
braise             13       1      3      0     15     0     2     39      
fried              34       0     39      0     37     0     1     54       
glaze              23       0      5      4     13     0     5     13       
grill             107       1      8      4    302     3    17    368       
mash               26       0      1      0     13     1     0     15        
none               32       0      5      1     40     0    12     35       
roast             129       2      4      8    173     0     5    818       
sautee              5       0      0      0     29     0     0     28       
scramble            5       0      0      0     15     0     2      4        
steam              12       0      2      1     13     0     1     19      

Predicted Style  scramble  steam  
Actual Style                      
bake                    0      4  
braise                  0      0  
fried                   1      0  
glaze                   0      1  
grill                   1      6  
mash                    0      0  
none                    0      0  
roast                   0      3  
sautee                  0      1  
scramble                1      0  
steam                   0      0

Random Forest
Cooking time was a stronger signal than presence of flour in the ingredients list.  

Feature importances:
[('has_flour', 0.30), ('totalTimeInHours', 0.70)]


Logistic Regression
Logistic Regression was run with four values for L1. L1 is also known as least absolute deviation and it minimizes the total absolute value of error (vs L2 in linear regression minimizing total square error). As C, the value for L1, decreases the model coefficients become smaller and the coefficients approach zero. The regularization penalty becomes more prominent.

For two features (cook time and flour) test accuracy was similar (.39) for C = 10, 1, and .1 but worse for C = .01. With additional engineered features: has_cheese, has_oil, has_seasoning, test accuracy was .44 for C = 10.


# Appendix

EXAMPLE OUTPUT (Logistic Regression):

('C:', 10)
('Coefficient of each feature:', array([[ 1.20815761,  0.02383756],
       [-0.31217111,  0.06299036],
       [ 1.92652442, -0.3360049 ],
       [ 2.19933023,  0.026496  ],
       [-1.57443222, -0.1569505 ],
       [-0.05036472, -0.38877279],
       [-1.70775676,  0.0358586 ],
       [-1.46492061, -3.61389425],
       [-1.51906797,  0.06156078],
       [ 0.12147283, -1.03967401]]))
('Training accuracy:', 0.3860015929908403)
('Test accuracy:', 0.39008363201911589)

('C:', 1)
('Coefficient of each feature:', array([[ 1.20376619,  0.02354267],
       [-0.26297252,  0.0624033 ],
       [ 1.91424595, -0.33145655],
       [ 2.17032768,  0.02406243],
       [-1.56168647, -0.15663481],
       [ 0.        , -0.38054964],
       [-1.69844933,  0.03549928],
       [-1.26817173, -3.56164817],
       [-1.16187896,  0.06043077],
       [ 0.04531187, -1.00787366]]))
('Training accuracy:', 0.3860015929908403)
('Test accuracy:', 0.39008363201911589)

('C:', 0.1)
('Coefficient of each feature:', array([[ 1.15996402,  0.02072297],
       [ 0.        ,  0.05616472],
       [ 1.79211938, -0.29010913],
       [ 1.8951356 ,  0.        ],
       [-1.44108941, -0.15445663],
       [ 0.        , -0.28900445],
       [-1.60807666,  0.03212333],
       [-0.2559184 , -3.08822341],
       [ 0.        ,  0.04607488],
       [ 0.        , -0.77473685]]))
('Training accuracy:', 0.3860015929908403)
('Test accuracy:', 0.39008363201911589)

('C:', 0.001)
('Coefficient of each feature:', array([[ 0.        ,  0.        ],
       [ 0.        ,  0.        ],
       [ 0.        ,  0.        ],
       [ 0.        ,  0.        ],
       [ 0.        , -0.02625996],
       [ 0.        ,  0.        ],
       [ 0.        ,  0.        ],
       [ 0.        ,  0.        ],
       [ 0.        ,  0.        ],
       [ 0.        ,  0.        ]]))
('Training accuracy:', 0.34099960175228994)
('Test accuracy:', 0.34109916367980886)


TFIDF with “roasted”, “grilled” and “steamed”
('cosine_similarities',
array([[ 1.        ,  0.0521508 ,  0.27907206],
       [ 0.0521508 ,  1.        ,  0.19912935],
       [ 0.27907206,  0.19912935,  1.        ]]))


Distribution of recipes per style word
4617 roast
3599 bake
3300 grill
669 fried
295 braise
261 glaze
259 sautee
231 mash
191 steam
109 scramble
88 poach
52 boil
10 puree
25 fry
7 gratin
2 blanch
