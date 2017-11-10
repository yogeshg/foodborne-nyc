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
# * data was received in the form of `Twitter_Labeled_2_10_17.xlsx`
# * this was saved in `twitter_sick_data.csv` in the same schema as yelp data
# 
# These data have the following issues, which we remedy below:
# * this data generally does not contain mentions of restaurants, so no steps have been taken to remedy that
#     

# ### Load in the sick and multiple data, removing the reviews that are different or are duplicate

# In[99]:
datadir = 'data/'
twitter_sick_datapath = datadir + 'twitter_sick_data.csv'

sick_df = pd.read_csv(twitter_sick_datapath)

logger.debug('length of sick dataframe: {}'.format(len(sick_df)))

# In[131]:

sick_data = {'x':np.array(list(sick_df['data'])),
             'y':np.array(list(sick_df['label'])),
             'old_sick_score':np.array(list(sick_df['old_score'])) }


# assuming no data could be restaurant-based


# split up the data into train and test using class-stratified sampling
def split_dev_test(data, test_size=.2):
    """ Get a stratified random sample of reviews, dropping any biased teset reviews """
    for train, test in cross_validation.StratifiedShuffleSplit(data['y'], n_iter=1, test_size=test_size, random_state=0):
        train_data = {k:v[train] for k,v in data.items()}
        test_data = {k:v[test] for k,v in data.items()}
        
    logger.info("Training/Dev data shape, x: {}, y: {} ".format(str(train_data['x'].shape), str(train_data['y'].shape)))
    logger.info("Test data shape x: {}, y: {} ".format( str(test_data['x'].shape), str(test_data['y'].shape)))
    return train_data, test_data

logger.info("Preparing Sick data:")
sick_train, sick_test = split_dev_test(sick_data)

