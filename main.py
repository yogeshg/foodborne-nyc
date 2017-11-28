import logging
logging.basicConfig(level = logging.INFO, format=
        '%(asctime)s:%(levelname)s:%(name)s:%(threadName)s:line %(lineno)d: %(message)s')
logger = logging.getLogger(__name__)

# Libraries
import sys
import json
import pandas as pd
# import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Source
import util
commit_hash = util.save_code()
from util.archiver import get_archiver
import config as c

# ML Libraries
# from keras.models import Model

from torch.nn import Module

# ML Source
from datasets import load
from datasets.experiments.baseline_experiment_util import importance_weighted_precision_recall

import models
import train
# from train import EarlyStopping, ModelCheckpoint, CSVLogger

# Global Variables
embeddings_matrix = None
X_train = None
y_train = None
w_train = None
z_train = None
X_test = None
y_test = None
w_test = None
z_test = None

def load_data(dataset, indexpath, embeddingspath):
    global embeddings_matrix, X_train, y_train, w_train, z_train, X_test, y_test, w_test, z_test
    
    embeddings_matrix = load.load_embeddings_matrix(indexpath, embeddingspath)
    ((X_train, y_train, w_train, z_train),
        (X_test, y_test, w_test, z_test), _) = load.load_devset_testset_index(dataset, indexpath)

    for x in (X_train, y_train, w_train, X_test, y_test, w_test):
        logger.debug("shape and info: "+str((x.shape, x.max(), x.min())))


def save_model(hyperparams, model, get_filename):
    '''
    hyperparams : dict of hyper parameters
    model : keras Model or pytorch Module
    get_filename : a function/or lambda that takes in a filename and retuns saveable path
    '''
    util.assert_type(hyperparams, dict)
    util.assert_type(model, Module)
    assert callable(get_filename), 'takes in a filename and retuns saveable path'

    with open(get_filename('hyperparameters.json'), 'w') as f:
        json.dump(hyperparams, f, sort_keys=True, indent=2)

    with open(get_filename('model.json'), 'w') as f:
        f.write(model.to_json(indent=2))

    stdout = sys.stdout
    with open(get_filename('summary.txt'), 'w') as sys.stdout:
        if isinstance(model, Module):
            sys.stdout.write(str(model))
    sys.stdout = stdout

    # util.plot_model(model, to_file=get_filename('model.png'), show_shapes=True, show_layer_names=True)

    return

def save_history(history, dirpath):
    '''
    Saves the parameters of training as returned by the history object of keras
    Saves the history dataframe, not required since also saved by csvlogger
    Plots the metrics required from this data, this depends on the experiment
    '''
    with open(dirpath+'/training.json', 'w') as f:
        json.dump(history.params, f, indent=2)

    df = pd.DataFrame.from_dict(history.history)
    df.to_csv(dirpath+'/history.csv')
    i = df.loc[:, c.monitor].argmax()

    for m in c.metrics + ['loss']:
        util.plot_metric(df, m, i, dirpath)

    return

# main
def run_experiments(finetune, kernel_sizes, filters, lr, pooling, kernel_l2_regularization, other_params):
    global embeddings_matrix, X_train, y_train, w_train, z_train
    #removing global X_test, y_test, w_test, z_test

    other_params['commit_hash'] = commit_hash

    maxlen = X_train.shape[1]
    (vocab_size, dimensions) = embeddings_matrix.shape
    net = models.Net(
        maxlen = maxlen, dimensions = dimensions, finetune = finetune, vocab_size = vocab_size,
        kernel_sizes = kernel_sizes, filters = filters,
        dropout_rate = 0.5, kernel_l2_regularization = kernel_l2_regularization, lr=lr,
        embeddings_matrix = embeddings_matrix)

    hyperparams = util.fill_dict(net.hyperparameters, other_params)

    with get_archiver(datadir='data/models', suffix="_"+commit_hash[:6]) as a1, get_archiver(datadir='data/results', suffix="_"+commit_hash[:6]) as a:

        save_model(hyperparams, net, a.getFilePath)

        early_stopping = train.EarlyStopping(importance_weighted_precision_recall, c.patience, c.monitor_objective)
        model_checkpoint = train.ModelCheckpoint(a1.getFilePath('weights.hdf5'), True, c.monitor_objective)
        csv_logger = train.CSVLogger(a.getFilePath('logger.csv'))

        adam_config = train.AdamConfig(lr=net.hyperparameters['lr'], beta_1=net.hyperparameters['beta_1'],
                                       beta_2=net.hyperparameters['beta_2'], epsilon=net.hyperparameters['epsilon'],
                                       decay=net.hyperparameters['decay'])

        history = train.fit(net, X_train, y_train, w_train, z_train,
                            batch_size=c.batch_size, epochs=c.epochs, validation_split=0.2,
                            callbacks = [early_stopping, model_checkpoint, csv_logger], optimizer=adam_config)

        save_history(history, a.getDirPath())

    return

