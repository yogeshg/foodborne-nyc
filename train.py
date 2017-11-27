from collections import namedtuple

import pandas as pd

import logging
logger = logging.getLogger(__name__)

EarlyStopping = namedtuple('EarlyStopping', ['monitor_func', 'patience', 'mode'])
ModelCheckpoint = namedtuple('ModelCheckpoint', ['file_path', 'save_best_only', 'mode'])
CSVLogger = namedtuple('CSVLogger', ['file_path'])

FitReturn = namedtuple('FitReturn', ['history', 'params'])

def fit(*args, **kwargs):
    logger.info('fitting with arguments: ' + str(args) + ', ' + str(kwargs))

    history = pd.DataFrame({col:[0] for col in 'loss,val_loss,acc,val_acc,auc,val_auc'.split(',')})
    params = {}

    return FitReturn(history, params)
