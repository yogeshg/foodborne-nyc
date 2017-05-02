# adapted from:
# FoodborneNYC/jamia_2017/official/Official Reproducible Results.ipynb
# https://github.com/teffland/FoodborneNYC.git

# coding: utf-8

# # Discovering Foodborne Illness in Online Restaurant Reviews - Official Experiments File
# 
# This notebook walks through all experiments presented in the manuscript "Discovering Foodborne Illness in Online Restaurant Reviews." Since there are many files for the original experiments and they are very hard to follow, this notebook consolidates all results and experiments presented in the manuscript in one linearly runnable notebook.
# 
# For questions about this notebook, please contact Tom Effland at teffland@.cs.columbia.edu
# 
# Notes:
# * This notebook does not include hyperparam tuning experiments, only the best found setting of parameters.  Those exploratory experiments can be found in `unofficial/Notebooks`. (Note they are _very_ difficult to follow)

# In[225]:

# modules
import csv

# libraries
import numpy as np
import numpy.random as npr
import pandas as pd
from sklearn import cross_validation
import logging
logger = logging.getLogger(__name__)

# # Data Preprocessing
# 
# `all_data_from_dohmh.xlsx` contains the original annotations from the DOHMH epidemiologists
# * This file contains multiple sheets with mixed schemas, with many missing labels or scores (basically it was unusable)
# * To remedy this, Anna pulled out (by hand) the first (chronological) 1,392 (1/3) of the about 5,000 reviews that had good scores and annotations. These were put into `yelp_sick_data.csv` and `yelp_mult_data.csv`.
# * These two files contain (almost) the same reviews, but with different labels for the Sick and Multiple tasks, respectively. 
# * To train and test the models, we perform a stratified (by label) sampling of the reviews into 80% train, 20% test 
# 
# These data have the following issues, which we remedy below:
# 
# * There each have 1 review that doesn't appear in the other file
#     * We remove both of these reviews. Data size 1392 -> 1391
# * There are a few duplicates for some inexplicable reason
#     * We remove all duplicates. Data size 1391 -> 1384
# * The train/test splits are not stratified by restaurant and so there is a potential restaurant label bias
#     * We remove all reviews from the test data that have a restaurant with at least one other review in the training data which shares the same label, for each task. This way no restaurant-specific features can be discriminative of class labels in the test set. Since the restaurant names were not saved during Anna's transfer, we look them up in the original csv. Data size: Sick: 1384 -> 1337, Multiple: 1384 -> 1333
#     

# ### Load in the sick and multiple data, removing the reviews that are different or are duplicate

# In[99]:
datadir = '/tmp/yo/foodborne/'
yelp_sick_datapath = datadir + 'yelp_sick_data.csv'
yelp_mult_datapath = datadir + 'yelp_mult_data.csv'
labelled_datapath = datadir + 'all_data_from_dohmh.xlsx'

sick_df = pd.read_csv(yelp_sick_datapath)
mult_df = pd.read_csv(yelp_mult_datapath)
logger.debug('length of sick dataframe: {} and mult: {}'.format(len(sick_df), len(mult_df)))


# In[97]:

sick_review_set = set(sick_df['data'].tolist())
mult_review_set = set(mult_df['data'].tolist())
mismatch_review_set = sick_review_set ^ mult_review_set
# print len(mismatch_review_set)
# print mismatch_review_set
all_data = {}
for _, row in sick_df.iterrows():
    if row['data'] not in mismatch_review_set:
        all_data[row['data']] = {'sick_label':row['label'],
                                   'old_sick_score':row['old_score']}
    else:
        pass #print row
for _, row in mult_df.iterrows():
    if row['data'] not in mismatch_review_set:
        all_data[row['data']].update({'mult_label':row['label'],
                                      'old_mult_score':row['old_score']})
    else:
        pass #print row
    
logger.info('Resulting number of reviews after mismatches and duplicates removed: {}'.format(len(all_data)))


# In[131]:

sick_data = {'x':np.array(all_data.keys()), 
             'y':np.array([d['sick_label'] for d in all_data.values()]),
             'old_sick_score':np.array([d['old_sick_score'] for d in all_data.values()])}
mult_data = {'x':np.array(all_data.keys()), 
             'y':np.array([d['mult_label'] for d in all_data.values()]),
             'old_mult_score':np.array([d['old_mult_score'] for d in all_data.values()])}


# In[311]:

logger.debug('{} positive multiple instances'.format(np.sum(mult_data['y'])))


# ### Now split into train and test sets, removing all data from the test sets that could be restaurant-biased. These are reviews that have a parent restaurant which has another review in the train data with the same label

# In[132]:

# compile mapping of all reviews to restaurants
xls = pd.ExcelFile(labelled_datapath)
df1 = xls.parse('allreviews')
df2 = xls.parse('July 12-Mar 13')
df3 = xls.parse('May 2013-present')
all_reviews = df1['Review'].tolist() + df2['Review'].tolist() + df3['Review'].tolist()

review2restaurant = {r:b for r,b in zip(df1['Review'].tolist(), df1['Business'].tolist())}
review2restaurant.update({r:b for r,b in zip(df2['Review'].tolist(), df2['Business'].tolist())})
review2restaurant.update({r:b for r,b in zip(df3['Review'].tolist(), df3['Business'].tolist())})


# In[178]:

# split up the data into train and test using class-stratified sampling
def split_dev_test(data, test_size=.2):
    """ Get a stratified random sample of reviews, dropping any biased teset reviews """
    restaurant_labelsets = {v: set([]) for v in review2restaurant.values()+['UNKNOWN']}
    for train, test in cross_validation.StratifiedShuffleSplit(data['y'], n_iter=1, test_size=test_size, random_state=0):
        train_data = {k:v[train] for k,v in data.items()}
        # get the restaurant of each review and add this review's label to its training label set
        for i in train:
            if data['x'][i] in review2restaurant:
                restaurant_labelsets[review2restaurant[data['x'][i]]] |= set([data['y'][i]])
            else: 
                # a few are missing the restaurant
                # this is likely an encoding error
                # but we take a conservative approach
                # by calling these all 'UNKNOWN' 
                # which causes all unknowns in test to be dropped
                restaurant_labelsets['UNKNOWN'] |= set([data['y'][i]])
                
        # for each test document
        # make sure that its restaurant does not have a review with the same label in the training set
        # if it does, then omit it from the test data
        good_idxs = []
        for i in test:
            review_restaurant = review2restaurant[data['x'][i]] if data['x'][i] in review2restaurant else 'UNKNOWN'
            if data['y'][i] not in restaurant_labelsets[review_restaurant]:
                good_idxs.append(i)
        good_idxs = np.array(good_idxs)
        test_data = {k:v[good_idxs] for k,v in data.items()}
        
        
    logger.info("Training/Dev data shape, x: {}, y: {} ".format(str(train_data['x'].shape), str(train_data['y'].shape)))
    logger.info("Test data shape x: {}, y: {} ".format( str(test_data['x'].shape), str(test_data['y'].shape)))
    return train_data, test_data

logger.info("Preparing Sick data:")
sick_train, sick_test = split_dev_test(sick_data)
logger.info("Preparing Mult data:")
mult_train, mult_test = split_dev_test(mult_data)

