
# Twitter runs for 'Discovering Foodborne Illness in Online Restaurant Reviews'

This notebook presents the evaluation of twitter classifiers presented in the paper.

## Table of Contents


1. [Setup](#)
    1. [Data Ingestion](#) (Data is cleaned ahead of time)
    4. [Loading Models](#) (Model hyperparams are pre-tuned)
2. [Sick Task](#)
    1. [Logistic Regression](#)
    2. [Random Forest](#)
    3. [SVM](#)
    4. [Prototype](#)
3. [Multiple Task](#)
    1. [Logistic Regression](#)
    2. [Random Forest](#)
    3. [SVM](#)
    4. [Pipelined Logistic Regression](#)
    5. [Prototype](#)

# Setup


```python
import time
import os
import json
%matplotlib inline
from copy import deepcopy
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter('ignore')
from __future__ import print_function
import logging
logging.basicConfig(level=logging.DEBUG)
```


```python
import torch
from torch.autograd import Variable
import main
import models
import inspector
import datasets.experiments.baseline_experiment_util as beutil
```


```python
random_seed = 0
all_results = {}
```


```python
selected_models = {
"twitter.gold"   : "data/models/20171217_020721_811949/",
"twitter.silver" : "data/models/20171217_175028_811949/",
"twitter.biased" : "data/models/20171217_022127_811949/",
"yelp.gold"      : "data/models/20171217_061943_811949/",
"yelp.silver"    : "data/models/20171217_195647_811949/",
"yelp.biased"    : "data/models/20171217_203244_811949/",
}

reload(inspector)

def load_net_make_inspector(dataset, embeddings_matrix):
    model_dirname = selected_models[dataset]
    net = torch.load(os.path.join(model_dirname, 'checkpoint.net.pt'), map_location=lambda storage,y: storage)
    print(net.embeddings)
    net_inspector = inspector.Inspector(net, embeddings_matrix)
    net_inspector.net.eval()
    print(net_inspector.net.embeddings)
    return net_inspector


def make_model_report(media, regime, net_inspector, test_data, B):
    start = time.time()
    title = " ".join((media, regime))
    save_fname = "figures/" + ("_".join((media, regime))).lower()
    all_results[title] = beutil.model_report(net_inspector, title, 'is_foodborne',
                                      test_data=test_data,
                                      save_fname=save_fname,
                                      B=B, random_seed=random_seed)
    print('\n{} seconds for evaluation'.format(int(time.time()-start)))

    
def print_model_hyperparams(name, net):
    print(name)
    print("-"*len(name))
    print(net)
    print(json.dumps(net.hyperparameters, indent=2))

```

## Twitter
Here are the notable stats for testing on the Twitter sick classification task:

### TODO!!!
* All the test data is from 1/1/2017 and later
* It's about 2/3 biased and 1/3 nonbiased (1975 and 1000 reviews, respectively)
* All 1000 nonbiased reviews are have `No` labels
* The 1975 biased reviews are about 52% `Yes`/`No` (1026/949)

### Data Ingestion


```python
main.load_data('twitter.gold', 'data/twitter_data/', 'data/glove.twitter.27B.200d.txt')
gold_data = beutil.setup_baseline_data(dataset='twitter', data_path='./data/twitter_data/',
                                test_regime='gold', train_regime='gold')

```

    DEBUG:pip.utils:lzma module is not available
    DEBUG:pip.vcs:Registered VCS backend: git
    DEBUG:pip.vcs:Registered VCS backend: hg
    DEBUG:pip.vcs:Registered VCS backend: svn
    DEBUG:pip.vcs:Registered VCS backend: bzr
    INFO:datasets.load:loading data for twitter.gold from data/twitter_data/
    INFO:datasets.experiments.baseline_experiment_util:data setup with len(train_data.text) = 8129 len(test_data.text) = 2822 all_B_over_U = 0.00173566547282
    INFO:datasets.load:label bias tuples count:1.0,0.0: 5494
    INFO:datasets.load:label bias tuples count:0.0,0.0: 875
    INFO:datasets.load:label bias tuples count:0.0,1.0: 10
    INFO:datasets.load:label bias tuples count:1.0,1.0: 1750
    INFO:root:shape of training data (X, y, w, z): ((7316, 36), (7316,), (7316,), (7316,))
    INFO:root:shape of validation data (X, y, w, z): ((813, 36), (813,), (813,), (813,))
    INFO:root:shape of testing data (X, y, w, z): ((2822, 36), (2822,), (2822,), (2822,))
    INFO:root:length of index2tokens: 2063
    INFO:datasets.load:embeddings found for 1978/2063 tokens, not found for 85
    INFO:root:shape of embeddings_matrix: (2063, 200)
    INFO:datasets.experiments.baseline_experiment_util:data setup with len(train_data.text) = 8129 len(test_data.text) = 2822 all_B_over_U = 0.00173566547282



```python
test_data = gold_data['test_data']
B = 1000 # number of bootstrap test set resamples

## replace the keys that are replacable in test_data with the main data
## since this has the data in required form for prediction
# print(main.testing_set.X.max())
test_data['text'] = main.testing_set.X

# print(len(test_data['is_foodborne']))
# print(main.testing_set.y.shape)
test_data['is_foodborne'] = main.testing_set.y

# print(len(test_data['is_biased'] ))
# print(main.testing_set.w.shape)
# print(main.testing_set.w.max(), main.testing_set.w.min())
# test_data['is_biased'] = main.testing_set.w
# --> 321     nonbiased_idxs = np.argwhere(~is_biased).ravel()
#     322     print
#     323     for i in range(B):
# TypeError: ufunc 'invert' not supported for the input types, and the inputs could
# not be safely coerced to any supported types according to the casting rule ''safe''

```

### Models Load


```python
twitter_biased_inspector = load_net_make_inspector('twitter.biased', main.embeddings_matrix)

twitter_silver_inspector =  load_net_make_inspector('twitter.silver', main.embeddings_matrix)

twitter_gold_inspector = load_net_make_inspector('twitter.gold', main.embeddings_matrix)

```

    Embedding(1783, 200)
    Embedding(2063, 200)
    Embedding(2081, 200)
    Embedding(2063, 200)
    Embedding(2063, 200)
    Embedding(2063, 200)



```python
make_model_report('Twitter', 'Biased', twitter_biased_inspector, test_data, B)
```

    DEBUG:inspector:predict on vector of type <type 'numpy.ndarray'>
    DEBUG:inspector:predict_proba on vector of type <type 'numpy.ndarray'> and shape (2822, 36)
    DEBUG:inspector:0: processed batch, shape of logits: (64, 1)
    DEBUG:inspector:shape of class_probs: (2822, 2)
    DEBUG:inspector:shape of predictions: (2822,)
    DEBUG:inspector:predict_proba on vector of type <type 'numpy.ndarray'> and shape (2822, 36)
    DEBUG:inspector:0: processed batch, shape of logits: (64, 1)
    DEBUG:inspector:shape of class_probs: (2822, 2)


    
    B: 999/1000         
    B: 999/1000 
    B: 999/1000  
    3 seconds for evaluation
    



![png](Official%20Results_files/Official%20Results_13_2.png)



![png](Official%20Results_files/Official%20Results_13_3.png)



```python
make_model_report('Twitter', 'Gold', twitter_gold_inspector, test_data, B)
```

    DEBUG:inspector:predict on vector of type <type 'numpy.ndarray'>
    DEBUG:inspector:predict_proba on vector of type <type 'numpy.ndarray'> and shape (2822, 36)
    DEBUG:inspector:0: processed batch, shape of logits: (64, 1)
    DEBUG:inspector:shape of class_probs: (2822, 2)
    DEBUG:inspector:shape of predictions: (2822,)
    DEBUG:inspector:predict_proba on vector of type <type 'numpy.ndarray'> and shape (2822, 36)
    DEBUG:inspector:0: processed batch, shape of logits: (64, 1)
    DEBUG:inspector:shape of class_probs: (2822, 2)


    
    B: 999/1000          
    B: 999/1000 
    B: 999/1000 
    3 seconds for evaluation
    



![png](Official%20Results_files/Official%20Results_14_2.png)



![png](Official%20Results_files/Official%20Results_14_3.png)



```python
make_model_report('Twitter', 'Silver', twitter_silver_inspector, test_data, B)
```

    DEBUG:inspector:predict on vector of type <type 'numpy.ndarray'>
    DEBUG:inspector:predict_proba on vector of type <type 'numpy.ndarray'> and shape (2822, 36)
    DEBUG:inspector:0: processed batch, shape of logits: (64, 1)
    DEBUG:inspector:shape of class_probs: (2822, 2)
    DEBUG:inspector:shape of predictions: (2822,)
    DEBUG:inspector:predict_proba on vector of type <type 'numpy.ndarray'> and shape (2822, 36)
    DEBUG:inspector:0: processed batch, shape of logits: (64, 1)
    DEBUG:inspector:shape of class_probs: (2822, 2)


    
    B: 999/1000          
    B: 999/1000 
    B: 999/1000 
    3 seconds for evaluation
    



![png](Official%20Results_files/Official%20Results_15_2.png)



![png](Official%20Results_files/Official%20Results_15_3.png)


### Precision Recall Tradeoffs

We'd like to explore how we lose precision with the LR models, as we gain recall. This can be visualized by looking at the high recall region of the PR curves. 

In the curve we can see that all of the model precision begins to drop around a recall of .8 start to significantly drop precision around a recall of .9


```python
beutil.pr_curves([twitter_biased_inspector, twitter_gold_inspector, twitter_silver_inspector], 
          ['Twitter Biased', 
           'Twitter Gold', 
           'Twitter Silver'], 
          'Precision-Recall Tradeoffs', 'is_foodborne', 
          dashes=[[20,5], [10,3], [5,1]],
          test_data=test_data, save_fname='figures/paper_twitter',
          figsize=(6,4),
          xlim=(.5,1.),
          yticks=.1*np.arange(11))
```

    DEBUG:inspector:predict on vector of type <type 'numpy.ndarray'>
    DEBUG:inspector:predict_proba on vector of type <type 'numpy.ndarray'> and shape (2822, 36)
    DEBUG:inspector:0: processed batch, shape of logits: (64, 1)
    DEBUG:inspector:shape of class_probs: (2822, 2)
    DEBUG:inspector:shape of predictions: (2822,)
    DEBUG:inspector:predict_proba on vector of type <type 'numpy.ndarray'> and shape (2822, 36)
    DEBUG:inspector:0: processed batch, shape of logits: (64, 1)
    DEBUG:inspector:shape of class_probs: (2822, 2)
    DEBUG:inspector:predict on vector of type <type 'numpy.ndarray'>
    DEBUG:inspector:predict_proba on vector of type <type 'numpy.ndarray'> and shape (2822, 36)
    DEBUG:inspector:0: processed batch, shape of logits: (64, 1)
    DEBUG:inspector:shape of class_probs: (2822, 2)
    DEBUG:inspector:shape of predictions: (2822,)
    DEBUG:inspector:predict_proba on vector of type <type 'numpy.ndarray'> and shape (2822, 36)
    DEBUG:inspector:0: processed batch, shape of logits: (64, 1)
    DEBUG:inspector:shape of class_probs: (2822, 2)
    DEBUG:inspector:predict on vector of type <type 'numpy.ndarray'>
    DEBUG:inspector:predict_proba on vector of type <type 'numpy.ndarray'> and shape (2822, 36)
    DEBUG:inspector:0: processed batch, shape of logits: (64, 1)
    DEBUG:inspector:shape of class_probs: (2822, 2)
    DEBUG:inspector:shape of predictions: (2822,)
    DEBUG:inspector:predict_proba on vector of type <type 'numpy.ndarray'> and shape (2822, 36)
    DEBUG:inspector:0: processed batch, shape of logits: (64, 1)
    DEBUG:inspector:shape of class_probs: (2822, 2)





    (<matplotlib.figure.Figure at 0x7fe6ceae8e50>,
     <matplotlib.axes._subplots.AxesSubplot at 0x7fe6cbfb7110>)




![png](Official%20Results_files/Official%20Results_17_2.png)



```python
print_model_hyperparams('Twitter Biased', twitter_biased_inspector.net)
print()
print_model_hyperparams('Twitter Gold', twitter_gold_inspector.net)
print()
print_model_hyperparams('Twitter Silver', twitter_silver_inspector.net)
print()
```

    Twitter Biased
    --------------
    Net(
      (embeddings): Embedding(2063, 200)
      (pad1_0): ConstantPad1d((1, 0))
      (conv1_0): Conv1d (200, 75, kernel_size=(1,), stride=(1,))
      (drop1_0): Dropout(p=0.5)
      (pad1_1): ConstantPad1d((1, 1))
      (conv1_1): Conv1d (200, 75, kernel_size=(2,), stride=(1,))
      (drop1_1): Dropout(p=0.5)
      (fc): Linear(in_features=150, out_features=1)
    )
    all_parameters: [[412600], [], [15000, 75], [], [], [30000, 75], [], [150, 1]]
    trainable_parameters: 0
    {
      "beta_1": 0.9, 
      "kernel_sizes": [
        1, 
        2
      ], 
      "dropout_rate": 0.5, 
      "beta_2": 0.999, 
      "dimensions": 200, 
      "vocab_size": 1783, 
      "epsilon": 1e-08, 
      "activation": "relu", 
      "pooling": "max", 
      "lr": 0.001, 
      "filters": 75, 
      "weight_decay": 0.001, 
      "finetune": false
    }
    
    Twitter Gold
    ------------
    Net(
      (embeddings): Embedding(2063, 200)
      (pad1_0): ConstantPad1d((1, 0))
      (conv1_0): Conv1d (200, 75, kernel_size=(1,), stride=(1,))
      (drop1_0): Dropout(p=0.5)
      (pad1_1): ConstantPad1d((1, 1))
      (conv1_1): Conv1d (200, 75, kernel_size=(2,), stride=(1,))
      (drop1_1): Dropout(p=0.5)
      (fc): Linear(in_features=150, out_features=1)
    )
    all_parameters: [[412600], [], [15000, 75], [], [], [30000, 75], [], [150, 1]]
    trainable_parameters: 0
    {
      "beta_1": 0.9, 
      "kernel_sizes": [
        1, 
        2
      ], 
      "dropout_rate": 0.5, 
      "beta_2": 0.999, 
      "dimensions": 200, 
      "vocab_size": 2063, 
      "epsilon": 1e-08, 
      "activation": "relu", 
      "pooling": "max", 
      "lr": 0.001, 
      "filters": 75, 
      "weight_decay": 0.001, 
      "finetune": false
    }
    
    Twitter Silver
    --------------
    Net(
      (embeddings): Embedding(2063, 200)
      (pad1_0): ConstantPad1d((1, 0))
      (conv1_0): Conv1d (200, 100, kernel_size=(1,), stride=(1,))
      (drop1_0): Dropout(p=0.5)
      (pad1_1): ConstantPad1d((1, 1))
      (conv1_1): Conv1d (200, 100, kernel_size=(2,), stride=(1,))
      (drop1_1): Dropout(p=0.5)
      (pad1_2): ConstantPad1d((2, 1))
      (conv1_2): Conv1d (200, 100, kernel_size=(3,), stride=(1,))
      (drop1_2): Dropout(p=0.5)
      (pad1_3): ConstantPad1d((2, 2))
      (conv1_3): Conv1d (200, 100, kernel_size=(4,), stride=(1,))
      (drop1_3): Dropout(p=0.5)
      (pad1_4): ConstantPad1d((3, 2))
      (conv1_4): Conv1d (200, 100, kernel_size=(5,), stride=(1,))
      (drop1_4): Dropout(p=0.5)
      (fc): Linear(in_features=500, out_features=1)
    )
    all_parameters: [[412600], [], [20000, 100], [], [], [40000, 100], [], [], [60000, 100], [], [], [80000, 100], [], [], [100000, 100], [], [500, 1]]
    trainable_parameters: 0
    {
      "beta_1": 0.9, 
      "kernel_sizes": [
        1, 
        2, 
        3, 
        4, 
        5
      ], 
      "dropout_rate": 0.5, 
      "beta_2": 0.999, 
      "dimensions": 200, 
      "vocab_size": 2081, 
      "epsilon": 1e-08, 
      "activation": "relu", 
      "pooling": "max", 
      "lr": 0.001, 
      "filters": 100, 
      "weight_decay": 0.001, 
      "finetune": false
    }
    


## Yelp
Here are the notable stats for testing on the Yelp sick classification task:

### TODO!!!
* All the test data is from 1/1/2017 and later
* It's about 2/3 biased and 1/3 nonbiased (1975 and 1000 reviews, respectively)
* All 1000 nonbiased reviews are have `No` labels
* The 1975 biased reviews are about 52% `Yes`/`No` (1026/949)

### Data Ingestion


```python
main.load_data('yelp.gold', 'data/yelp_data/', 'data/glove.840B.300d.txt')
gold_data = beutil.setup_baseline_data(dataset='yelp', data_path='./data/yelp_data/',
                                test_regime='gold', train_regime='gold')

```

    INFO:datasets.load:loading data for yelp.gold from data/yelp_data/
    INFO:datasets.experiments.baseline_experiment_util:data setup with len(train_data.text) = 12566 len(test_data.text) = 2975 all_B_over_U = 0.0113831143885
    INFO:datasets.load:label bias tuples count:1.0,0.0: 5667
    INFO:datasets.load:label bias tuples count:0.0,0.0: 999
    INFO:datasets.load:label bias tuples count:0.0,1.0: 1
    INFO:datasets.load:label bias tuples count:1.0,1.0: 5899
    WARNING:datasets.load:found a class with 1 item only, trying to sparsify on biased alone
    INFO:root:shape of training data (X, y, w, z): ((11309, 857), (11309,), (11309,), (11309,))
    INFO:root:shape of validation data (X, y, w, z): ((1257, 857), (1257,), (1257,), (1257,))
    INFO:root:shape of testing data (X, y, w, z): ((2975, 857), (2975,), (2975,), (2975,))
    INFO:root:length of index2tokens: 9102
    INFO:datasets.load:embeddings found for 8932/9102 tokens, not found for 170
    INFO:root:shape of embeddings_matrix: (9102, 300)
    INFO:datasets.experiments.baseline_experiment_util:data setup with len(train_data.text) = 12566 len(test_data.text) = 2975 all_B_over_U = 0.0113831143885



```python
test_data = gold_data['test_data']
B = 1000 # number of bootstrap test set resamples

## replace the keys that are replacable in test_data with the main data
## since this has the data in required form for prediction
# print(main.testing_set.X.max())
test_data['text'] = main.testing_set.X

# print(len(test_data['is_foodborne']))
# print(main.testing_set.y.shape)
test_data['is_foodborne'] = main.testing_set.y

# print(len(test_data['is_biased'] ))
# print(main.testing_set.w.shape)
# print(main.testing_set.w.max(), main.testing_set.w.min())
# test_data['is_biased'] = main.testing_set.w
# --> 321     nonbiased_idxs = np.argwhere(~is_biased).ravel()
#     322     print
#     323     for i in range(B):
# TypeError: ufunc 'invert' not supported for the input types, and the inputs could
# not be safely coerced to any supported types according to the casting rule ''safe''

```

### Models Load


```python
yelp_biased_inspector = load_net_make_inspector('yelp.biased', main.embeddings_matrix)

yelp_silver_inspector =  load_net_make_inspector('yelp.silver', main.embeddings_matrix)

yelp_gold_inspector = load_net_make_inspector('yelp.gold', main.embeddings_matrix)

```

    Embedding(8581, 300)
    Embedding(9102, 300)
    Embedding(9066, 300)
    Embedding(9102, 300)
    Embedding(9102, 300)
    Embedding(9102, 300)



```python
make_model_report('Yelp', 'Biased', yelp_biased_inspector, test_data, B)
```

    DEBUG:inspector:predict on vector of type <type 'numpy.ndarray'>
    DEBUG:inspector:predict_proba on vector of type <type 'numpy.ndarray'> and shape (2975, 857)
    DEBUG:inspector:0: processed batch, shape of logits: (64, 1)
    DEBUG:inspector:shape of class_probs: (2975, 2)
    DEBUG:inspector:shape of predictions: (2975,)
    DEBUG:inspector:predict_proba on vector of type <type 'numpy.ndarray'> and shape (2975, 857)
    DEBUG:inspector:0: processed batch, shape of logits: (64, 1)
    DEBUG:inspector:shape of class_probs: (2975, 2)


    
    B: 999/1000               
    B: 999/1000 
    B: 999/1000 
    8 seconds for evaluation
    



![png](Official%20Results_files/Official%20Results_25_2.png)



![png](Official%20Results_files/Official%20Results_25_3.png)



```python
make_model_report('Yelp', 'Gold', yelp_gold_inspector, test_data, B)
```

    DEBUG:inspector:predict on vector of type <type 'numpy.ndarray'>
    DEBUG:inspector:predict_proba on vector of type <type 'numpy.ndarray'> and shape (2975, 857)
    DEBUG:inspector:0: processed batch, shape of logits: (64, 1)
    DEBUG:inspector:shape of class_probs: (2975, 2)
    DEBUG:inspector:shape of predictions: (2975,)
    DEBUG:inspector:predict_proba on vector of type <type 'numpy.ndarray'> and shape (2975, 857)
    DEBUG:inspector:0: processed batch, shape of logits: (64, 1)
    DEBUG:inspector:shape of class_probs: (2975, 2)


    
    B: 999/1000               
    B: 999/1000  
    B: 999/1000 
    7 seconds for evaluation
    



![png](Official%20Results_files/Official%20Results_26_2.png)



![png](Official%20Results_files/Official%20Results_26_3.png)



```python
make_model_report('Yelp', 'Silver', yelp_silver_inspector, test_data, B)
```

    DEBUG:inspector:predict on vector of type <type 'numpy.ndarray'>
    DEBUG:inspector:predict_proba on vector of type <type 'numpy.ndarray'> and shape (2975, 857)
    DEBUG:inspector:0: processed batch, shape of logits: (64, 1)
    DEBUG:inspector:shape of class_probs: (2975, 2)
    DEBUG:inspector:shape of predictions: (2975,)
    DEBUG:inspector:predict_proba on vector of type <type 'numpy.ndarray'> and shape (2975, 857)
    DEBUG:inspector:0: processed batch, shape of logits: (64, 1)
    DEBUG:inspector:shape of class_probs: (2975, 2)


    
    B: 999/1000               
    B: 999/1000  
    B: 999/1000  
    8 seconds for evaluation
    



![png](Official%20Results_files/Official%20Results_27_2.png)



![png](Official%20Results_files/Official%20Results_27_3.png)


### Precision Recall Tradeoffs

We'd like to explore how we lose precision with the LR models, as we gain recall. This can be visualized by looking at the high recall region of the PR curves. 

In the curve we can see that all of the model precision begins to drop around a recall of .8 start to significantly drop precision around a recall of .9


```python
beutil.pr_curves([yelp_biased_inspector, yelp_gold_inspector, yelp_silver_inspector], 
          ['Yelp Biased', 
           'Yelp Gold', 
           'Yelp Silver'], 
          'Precision-Recall Tradeoffs', 'is_foodborne', 
          dashes=[[20,5], [10,3], [5,1]],
          test_data=test_data, save_fname='figures/paper_yelp',
          figsize=(6,4),
          xlim=(.5,1.),
          yticks=.1*np.arange(11))
```

    DEBUG:inspector:predict on vector of type <type 'numpy.ndarray'>
    DEBUG:inspector:predict_proba on vector of type <type 'numpy.ndarray'> and shape (2975, 857)
    DEBUG:inspector:0: processed batch, shape of logits: (64, 1)
    DEBUG:inspector:shape of class_probs: (2975, 2)
    DEBUG:inspector:shape of predictions: (2975,)
    DEBUG:inspector:predict_proba on vector of type <type 'numpy.ndarray'> and shape (2975, 857)
    DEBUG:inspector:0: processed batch, shape of logits: (64, 1)
    DEBUG:inspector:shape of class_probs: (2975, 2)
    DEBUG:inspector:predict on vector of type <type 'numpy.ndarray'>
    DEBUG:inspector:predict_proba on vector of type <type 'numpy.ndarray'> and shape (2975, 857)
    DEBUG:inspector:0: processed batch, shape of logits: (64, 1)
    DEBUG:inspector:shape of class_probs: (2975, 2)
    DEBUG:inspector:shape of predictions: (2975,)
    DEBUG:inspector:predict_proba on vector of type <type 'numpy.ndarray'> and shape (2975, 857)
    DEBUG:inspector:0: processed batch, shape of logits: (64, 1)
    DEBUG:inspector:shape of class_probs: (2975, 2)
    DEBUG:inspector:predict on vector of type <type 'numpy.ndarray'>
    DEBUG:inspector:predict_proba on vector of type <type 'numpy.ndarray'> and shape (2975, 857)
    DEBUG:inspector:0: processed batch, shape of logits: (64, 1)
    DEBUG:inspector:shape of class_probs: (2975, 2)
    DEBUG:inspector:shape of predictions: (2975,)
    DEBUG:inspector:predict_proba on vector of type <type 'numpy.ndarray'> and shape (2975, 857)
    DEBUG:inspector:0: processed batch, shape of logits: (64, 1)
    DEBUG:inspector:shape of class_probs: (2975, 2)





    (<matplotlib.figure.Figure at 0x7fe6cdbdfb90>,
     <matplotlib.axes._subplots.AxesSubplot at 0x7fe6c6c77450>)




![png](Official%20Results_files/Official%20Results_29_2.png)



```python
print_model_hyperparams('Yelp Biased', yelp_biased_inspector.net)
print()
print_model_hyperparams('Yelp Gold', yelp_gold_inspector.net)
print()
print_model_hyperparams('Yelp Silver', yelp_silver_inspector.net)
print()
```

    Yelp Biased
    -----------
    Net(
      (embeddings): Embedding(9102, 300)
      (pad1_0): ConstantPad1d((1, 0))
      (conv1_0): Conv1d (300, 50, kernel_size=(1,), stride=(1,))
      (drop1_0): Dropout(p=0.5)
      (pad1_1): ConstantPad1d((1, 1))
      (conv1_1): Conv1d (300, 50, kernel_size=(2,), stride=(1,))
      (drop1_1): Dropout(p=0.5)
      (pad1_2): ConstantPad1d((2, 1))
      (conv1_2): Conv1d (300, 50, kernel_size=(3,), stride=(1,))
      (drop1_2): Dropout(p=0.5)
      (pad1_3): ConstantPad1d((2, 2))
      (conv1_3): Conv1d (300, 50, kernel_size=(4,), stride=(1,))
      (drop1_3): Dropout(p=0.5)
      (pad1_4): ConstantPad1d((3, 2))
      (conv1_4): Conv1d (300, 50, kernel_size=(5,), stride=(1,))
      (drop1_4): Dropout(p=0.5)
      (fc): Linear(in_features=250, out_features=1)
    )
    all_parameters: [[2730600], [], [15000, 50], [], [], [30000, 50], [], [], [45000, 50], [], [], [60000, 50], [], [], [75000, 50], [], [250, 1]]
    trainable_parameters: 0
    {
      "beta_1": 0.9, 
      "kernel_sizes": [
        1, 
        2, 
        3, 
        4, 
        5
      ], 
      "dropout_rate": 0.5, 
      "beta_2": 0.999, 
      "dimensions": 300, 
      "vocab_size": 8581, 
      "epsilon": 1e-08, 
      "activation": "relu", 
      "pooling": "max", 
      "lr": 0.001, 
      "filters": 50, 
      "weight_decay": 0.001, 
      "finetune": false
    }
    
    Yelp Gold
    ---------
    Net(
      (embeddings): Embedding(9102, 300)
      (pad1_0): ConstantPad1d((1, 0))
      (conv1_0): Conv1d (300, 20, kernel_size=(1,), stride=(1,))
      (drop1_0): Dropout(p=0.5)
      (pad1_1): ConstantPad1d((1, 1))
      (conv1_1): Conv1d (300, 20, kernel_size=(2,), stride=(1,))
      (drop1_1): Dropout(p=0.5)
      (pad1_2): ConstantPad1d((2, 1))
      (conv1_2): Conv1d (300, 20, kernel_size=(3,), stride=(1,))
      (drop1_2): Dropout(p=0.5)
      (pad1_3): ConstantPad1d((2, 2))
      (conv1_3): Conv1d (300, 20, kernel_size=(4,), stride=(1,))
      (drop1_3): Dropout(p=0.5)
      (pad1_4): ConstantPad1d((3, 2))
      (conv1_4): Conv1d (300, 20, kernel_size=(5,), stride=(1,))
      (drop1_4): Dropout(p=0.5)
      (fc): Linear(in_features=100, out_features=1)
    )
    all_parameters: [[2730600], [], [6000, 20], [], [], [12000, 20], [], [], [18000, 20], [], [], [24000, 20], [], [], [30000, 20], [], [100, 1]]
    trainable_parameters: 0
    {
      "beta_1": 0.9, 
      "kernel_sizes": [
        1, 
        2, 
        3, 
        4, 
        5
      ], 
      "dropout_rate": 0.5, 
      "beta_2": 0.999, 
      "dimensions": 300, 
      "vocab_size": 9102, 
      "epsilon": 1e-08, 
      "activation": "relu", 
      "pooling": "max", 
      "lr": 0.001, 
      "filters": 20, 
      "weight_decay": 0.001, 
      "finetune": false
    }
    
    Yelp Silver
    -----------
    Net(
      (embeddings): Embedding(9102, 300)
      (pad1_0): ConstantPad1d((1, 0))
      (conv1_0): Conv1d (300, 75, kernel_size=(1,), stride=(1,))
      (drop1_0): Dropout(p=0.5)
      (pad1_1): ConstantPad1d((1, 1))
      (conv1_1): Conv1d (300, 75, kernel_size=(2,), stride=(1,))
      (drop1_1): Dropout(p=0.5)
      (pad1_2): ConstantPad1d((2, 1))
      (conv1_2): Conv1d (300, 75, kernel_size=(3,), stride=(1,))
      (drop1_2): Dropout(p=0.5)
      (pad1_3): ConstantPad1d((2, 2))
      (conv1_3): Conv1d (300, 75, kernel_size=(4,), stride=(1,))
      (drop1_3): Dropout(p=0.5)
      (pad1_4): ConstantPad1d((3, 2))
      (conv1_4): Conv1d (300, 75, kernel_size=(5,), stride=(1,))
      (drop1_4): Dropout(p=0.5)
      (fc): Linear(in_features=375, out_features=1)
    )
    all_parameters: [[2730600], [], [22500, 75], [], [], [45000, 75], [], [], [67500, 75], [], [], [90000, 75], [], [], [112500, 75], [], [375, 1]]
    trainable_parameters: 0
    {
      "beta_1": 0.9, 
      "kernel_sizes": [
        1, 
        2, 
        3, 
        4, 
        5
      ], 
      "dropout_rate": 0.5, 
      "beta_2": 0.999, 
      "dimensions": 300, 
      "vocab_size": 9066, 
      "epsilon": 1e-08, 
      "activation": "relu", 
      "pooling": "max", 
      "lr": 0.001, 
      "filters": 75, 
      "weight_decay": 0.001, 
      "finetune": false
    }
    


## Compile the scores into a nice table


```python
sick_table = pd.DataFrame()
# mult_table = pd.DataFrame()
for name, result in sorted(all_results.items(), key=lambda x:x[0]):
    data = {k:v for k,v in result.items() if ('samples' not in k) and ('_ci' not in k) }
    data.update({k+'_b':v[0] for k,v in result.items() if '_ci' in k})
    data.update({k+'_t':v[1] for k,v in result.items() if '_ci' in k})
    data['name'] = name
    sick_table = sick_table.append(data, ignore_index=True)
    if 'Sick' in name and 'Sick Only' not in name:
        pass
    else:
        pass
        # mult_table = mult_table.append(data, ignore_index=True)
sick_table.set_index('name', inplace=True)
# mult_table.set_index('name', inplace=True)
```


```python
print(sick_table.columns)
```

    Index([u'aupr', u'aupr_ci_b', u'aupr_ci_t', u'biased_f1', u'biased_f1_ci_b',
           u'biased_f1_ci_t', u'biased_precision', u'biased_recall', u'mixed_f1',
           u'mixed_f1_ci_b', u'mixed_f1_ci_t', u'mixed_precision',
           u'mixed_recall'],
          dtype='object')



```python
sick_table.to_csv('sick_results.csv')
sick_table[['mixed_f1', 'mixed_f1_ci_b', 'mixed_f1_ci_t']]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mixed_f1</th>
      <th>mixed_f1_ci_b</th>
      <th>mixed_f1_ci_t</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Twitter Biased</th>
      <td>0.225854</td>
      <td>0.009666</td>
      <td>0.560058</td>
    </tr>
    <tr>
      <th>Twitter Gold</th>
      <td>0.462562</td>
      <td>0.049025</td>
      <td>0.757894</td>
    </tr>
    <tr>
      <th>Twitter Silver</th>
      <td>0.405223</td>
      <td>0.025976</td>
      <td>0.738011</td>
    </tr>
    <tr>
      <th>Yelp Biased</th>
      <td>0.748101</td>
      <td>0.724115</td>
      <td>0.773839</td>
    </tr>
    <tr>
      <th>Yelp Gold</th>
      <td>0.765721</td>
      <td>0.741847</td>
      <td>0.787329</td>
    </tr>
    <tr>
      <th>Yelp Silver</th>
      <td>0.729167</td>
      <td>0.707052</td>
      <td>0.751962</td>
    </tr>
  </tbody>
</table>
</div>




```python
sick_table
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>aupr</th>
      <th>aupr_ci_b</th>
      <th>aupr_ci_t</th>
      <th>biased_f1</th>
      <th>biased_f1_ci_b</th>
      <th>biased_f1_ci_t</th>
      <th>biased_precision</th>
      <th>biased_recall</th>
      <th>mixed_f1</th>
      <th>mixed_f1_ci_b</th>
      <th>mixed_f1_ci_t</th>
      <th>mixed_precision</th>
      <th>mixed_recall</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Twitter Biased</th>
      <td>0.317616</td>
      <td>0.012258</td>
      <td>0.650376</td>
      <td>0.320197</td>
      <td>0.261067</td>
      <td>0.378251</td>
      <td>0.500000</td>
      <td>0.235507</td>
      <td>0.225854</td>
      <td>0.009666</td>
      <td>0.560058</td>
      <td>0.950075</td>
      <td>0.128160</td>
    </tr>
    <tr>
      <th>Twitter Gold</th>
      <td>0.268128</td>
      <td>0.018070</td>
      <td>0.588397</td>
      <td>0.484935</td>
      <td>0.440106</td>
      <td>0.532951</td>
      <td>0.401425</td>
      <td>0.612319</td>
      <td>0.462562</td>
      <td>0.049025</td>
      <td>0.757894</td>
      <td>0.586690</td>
      <td>0.381786</td>
    </tr>
    <tr>
      <th>Twitter Silver</th>
      <td>0.315653</td>
      <td>0.026039</td>
      <td>0.631559</td>
      <td>0.519626</td>
      <td>0.467469</td>
      <td>0.564756</td>
      <td>0.536680</td>
      <td>0.503623</td>
      <td>0.405223</td>
      <td>0.025976</td>
      <td>0.738011</td>
      <td>0.953898</td>
      <td>0.257253</td>
    </tr>
    <tr>
      <th>Yelp Biased</th>
      <td>0.830340</td>
      <td>0.798251</td>
      <td>0.866625</td>
      <td>0.748101</td>
      <td>0.723830</td>
      <td>0.772673</td>
      <td>0.936609</td>
      <td>0.622761</td>
      <td>0.748101</td>
      <td>0.724115</td>
      <td>0.773839</td>
      <td>0.936609</td>
      <td>0.622761</td>
    </tr>
    <tr>
      <th>Yelp Gold</th>
      <td>0.815140</td>
      <td>0.780851</td>
      <td>0.852724</td>
      <td>0.765721</td>
      <td>0.742870</td>
      <td>0.788271</td>
      <td>0.922734</td>
      <td>0.654373</td>
      <td>0.765721</td>
      <td>0.741847</td>
      <td>0.787329</td>
      <td>0.922734</td>
      <td>0.654373</td>
    </tr>
    <tr>
      <th>Yelp Silver</th>
      <td>0.806089</td>
      <td>0.770910</td>
      <td>0.844834</td>
      <td>0.729167</td>
      <td>0.704255</td>
      <td>0.755251</td>
      <td>0.954003</td>
      <td>0.590095</td>
      <td>0.729167</td>
      <td>0.707052</td>
      <td>0.751962</td>
      <td>0.954003</td>
      <td>0.590095</td>
    </tr>
  </tbody>
</table>
</div>


