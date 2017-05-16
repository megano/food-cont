'''
Mongo commands

mongo
show dbs;
use recipesDB;
show collections;

db.createUser({
    user:"username",
    pwd:"passwordhere",
    roles: ["readWrite", "dbAdmin"]
});

# verify total # recipes.
db.recipes.count()

# check for match of single style word
db.recipes.find({ title: "grilled"})

# Count recipes that have any words in the style list in the recipe title.
db.recipes.find({recipeName:{$in:[ /^bake/i, /^scramble/i, /^fried/i, /^boil/i, /^roast/i, /^blanch/i, /^braise/i, /^puree/i, /^poach/i, /^grill/i, /^steam/i, /^mash/i, /^sautee/i, /^gratin/i, /^glaze/i]}}).count();

# count recipes with no style words.
db.recipes.find({recipeName:{$nin:[ /^bake/i, /^scramble/i, /^fried/i, /^boil/i, /^roast/i, /^blanch/i, /^braise/i, /^puree/i, /^poach/i, /^grill/i, /^steam/i, /^mash/i, /^sautee/i, /^gratin/i, /^glaze/i]}}).count();

# Tested with single word. This query exports records matching "baked" style word into style[stylewordhere].txt
mongoexport -d recipesDB -c recipes -q '{recipeName:{$in:[ /^bake/i]}}' --out exportdir/style-baked.json

#This query from command line exports all records matching any style word into a file called style-all.json
mongoexport -d recipesDB -c recipes -q '{recipeName:{$in:[ /^bake/i, /^scramble/i, /^fried/i, /^boil/i, /^roast/i, /^blanch/i, /^braise/i, /^puree/i, /^poach/i, /^grill/i, /^steam/i, /^mash/i, /^sautee/i, /^gratin/i, /^glaze/i]}}' --out exportdir/style-all.json

#This query from command line exports all records matching NO style word into a file called style-none.json
mongoexport -d recipesDB -c recipes -q '{recipeName:{$nin:[ /^bake/i, /^scramble/i, /^fried/i, /^boil/i, /^roast/i, /^blanch/i, /^braise/i, /^puree/i, /^poach/i, /^grill/i, /^steam/i, /^mash/i, /^sautee/i, /^gratin/i, /^glaze/i]}}' --out exportdir/style-none.json

'''
