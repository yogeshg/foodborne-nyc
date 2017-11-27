from collections import namedtuple

import pandas as pd

import numpy as np
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

    args.net.cuda()

    history = []
    try:
        for epoch in range(0, args.epochs):
            logger.info('starting epoch {} of {}'.format(epoch, args.epochs))

            epoch_loss = torch.FloatTensor(np.array([0.0])).cuda()
            epoch_num_correct = torch.LongTensor(np.array([0.0])).cuda()
            epoch_total_num = torch.LongTensor(np.array([0.0])).cuda()

            tt = lambda x: x.view((1, x.size()[0]))


            for i, ((X, y), (w, _)) in enumerate(zip(x_train_loader, w_train_loader)):

                # wrap inputs as variables
                X = Variable(X.cuda())
                y = Variable(y.cuda())
                w = Variable(w.cuda())

                # zero the parameter gradient
                adam.zero_grad()

                # forward + backward
                yp = args.net(X)
                logits = torch.cat((tt(1 - yp), tt(yp))).t()
                batch_loss_values = entropy_values(logits, y)
                batch_loss = (w.float() * batch_loss_values).sum()


                epoch_loss += batch_loss.data
                epoch_total_num += y.size()[0]
                epoch_num_correct += (tt(y).t() == (yp > 0.5).long()).long().sum().long().data

                batch_loss.backward()
                adam.step()


                #  Todo: plot histograms for yp, w, loss_values, loss

                # logger.debug('y.size()  :{}'.format(y.size()))
                # logger.debug('yp.size() :{}'.format(yp.size()))
                # logger.debug('w.size()  :{}'.format(w.size()))
                # logger.debug('entropy_values(yp, y).size()  :{}'.format(entropy_values(yp, y).size()))
                # logger.debug('w :{}'.format(w))
                # u.log_frequently(10, i,logger.info, '{}-th batch loss_values: {}'.format(i, tt(loss_values)))
                # u.log_frequently(10, i,logger.debug, '{}-th batch loss collected: {}'.format(i, epoch_loss))

            # logger.debug('epoch loss: {}'.format(epoch_loss))
            # logger.debug('epoch total: {}'.format(epoch_total_num))
            # logger.debug('epoch correct: {}'.format(epoch_num_correct))
            history.append([epoch_loss.cpu().numpy()[0],
                            float(epoch_num_correct.cpu().numpy()[0])/float(epoch_total_num.cpu().numpy()[0]),
                            float(epoch_total_num.cpu().numpy()[0])
                            ])

            for (X, y), (w, _) in zip(x_valid_loader, w_valid_loader):
                logger.debug('next valid batch size: ' + str(y.size()))
    except Exception, e:
        logger.exception(e)
        logger.info('error in training, continuing')

    params = {}
    # history_df = pd.DataFrame({col: [0] for col in 'loss,val_loss,acc,val_acc,auc,val_auc'.split(',')})
    history_df = pd.DataFrame.from_records(history)
    history_df.columns = ['loss', 'acc', 'auc']
    history_df.loc[:, 'val_loss'] = history_df.loc[:, 'loss']
    history_df.loc[:, 'val_acc'] = history_df.loc[:, 'acc']
    history_df.loc[:, 'val_auc'] = history_df.loc[:, 'auc']
    # history_df = pd.DataFrame({'loss': history,'val_loss','acc','val_acc','auc','val_auc'})

    return FitReturn(history_df, params)
