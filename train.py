from collections import namedtuple

import pandas as pd

import logging

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

import util as u

EarlyStopping = namedtuple('EarlyStopping', ['monitor_func', 'patience', 'mode'])
ModelCheckpoint = namedtuple('ModelCheckpoint', ['file_path', 'save_best_only', 'mode'])
CSVLogger = namedtuple('CSVLogger', ['file_path'])

FitArgs = namedtuple('FitArgs', ['net', 'X', 'y', 'w', 'batch_size', 'epochs', 'validation_split', 'callbacks', 'optimizer'])
AdamConfig = namedtuple('AdamConfig', ['lr', 'beta_1', 'beta_2', 'epsilon', 'decay'])
FitReturn = namedtuple('FitReturn', ['history', 'params'])


def get_loaders(x, y, w, start, end, batch_size):
    logger.debug('getting loader for matrices of size {x.shape[0]}, from {start} to {end}'.format(**locals()))
    assert x.shape[0] == y.shape[0] == w.shape[0]
    assert 0 <= start and start <= end and end <= x.shape[0]
    assert isinstance(start, int)
    assert isinstance(end, int)

    x_tensor = torch.LongTensor(x[start:end])
    y_tensor = torch.LongTensor(y[start:end])
    w_tensor = torch.FloatTensor(w[start:end])

    x_loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size)
    w_loader = DataLoader(TensorDataset(w_tensor, y_tensor), batch_size=batch_size)

    return x_loader, w_loader


# Note: loss is calculated per data point, then weighted by the weight
# vector. This is done based on the suggestions in following issue:
# https://github.com/pytorch/pytorch/issues/264
#

def fit(*pargs, **kwargs):
    args = FitArgs(*pargs, **kwargs)
    adam_config = args.optimizer
    logger.debug('fitting with arguments: ' + str(args))

    total_size = args.X.shape[0]
    train_size = int((total_size * (1.00 - args.validation_split)) // 1.0)

    x_train_loader, w_train_loader = get_loaders(args.X, args.y, args.w, 0, train_size, args.batch_size)
    x_valid_loader, w_valid_loader = get_loaders(args.X, args.y, args.w, train_size, total_size, args.batch_size)

    entropy_values = nn.CrossEntropyLoss(reduce=False)
    adam = optim.Adam(args.net.parameters(),
                      lr = adam_config.lr, betas = (adam_config.beta_1, adam_config.beta_2),
                      eps = adam_config.epsilon, weight_decay = adam_config.decay)

    for epoch in range(0, args.epochs):
        logger.info('starting epoch {} of {}'.format(epoch, args.epochs))

        epoch_loss = float(0.0)

        for i, ((X, y), (w, _)) in enumerate(zip(x_train_loader, w_train_loader)):

            # wrap inputs as variables
            X = Variable(X)
            y = Variable(y)
            w = Variable(w)

            # zero the parameter gradient
            adam.zero_grad()

            # forward + backward
            yp = args.net(X)
            loss = (w.float() * entropy_values(yp, y).float()).sum()
            loss.backward()
            adam.step()

            epoch_loss += loss.data.numpy()[0]

            #  Todo: plot histograms for yp, w, loss_values, loss

            # logger.debug('y.size()  :{}'.format(y.size()))
            # logger.debug('yp.size() :{}'.format(yp.size()))
            # logger.debug('w.size()  :{}'.format(w.size()))
            # logger.debug('entropy_values(yp, y).size()  :{}'.format(entropy_values(yp, y).size()))
            # logger.debug('w :{}'.format(w))
            u.log_frequently(10, i,logger.debug, '{}-th batch loss collected: {}'.format(i, epoch_loss))

        logger.debug('epoch loss: {}'.format(epoch_loss))

        for (X, y), (w, _) in zip(x_valid_loader, w_valid_loader):
            logger.debug('next valid batch size: ' + str(y.size()))

    history = pd.DataFrame({col: [0] for col in 'loss,val_loss,acc,val_acc,auc,val_auc'.split(',')})
    params = {}

    return FitReturn(history, params)
