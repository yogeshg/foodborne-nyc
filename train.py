from collections import namedtuple, OrderedDict

import pandas as pd
import os

import numpy as np
import logging

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

import datasets.load
import datasets.experiments.baseline_experiment_util as beutil
import util as u

DEBUG_TEST_PR_CALCULATION = True

EarlyStopping = namedtuple('EarlyStopping', ['monitor', 'patience', 'mode'])
ModelCheckpoint = namedtuple('ModelCheckpoint', ['file_path'])
CSVLogger = namedtuple('CSVLogger', ['file_path'])


class AfterEpoch():

    def __init__(self, early_stopping, model_checkpoint):
        """

        :param early_stopping: EarlyStopping
        :param model_checkpoint: ModelCheckpoint
        """
        self.early_stopping = early_stopping
        self.model_checkpoint = model_checkpoint
        self.num_checked = -1
        self.prev_best = np.inf if self.early_stopping.mode == 'min' else -np.inf
        self.prev_best_at = -1
        self.op = np.less if self.early_stopping.mode == 'min' else np.greater
        self.best_not_seen_for = 0

    def save_and_stop(self, record, net, optim):
        """

        :param record: dict
        :param net: nn.Module
        :param net: torch.optim
        :return:
        """

        val = record[self.early_stopping.monitor]
        self.num_checked += 1
        if self.op(val, self.prev_best):
            self.prev_best = val
            self.prev_best_at = self.num_checked
            logger.info("found new {} {} value {}".format(
                self.early_stopping.mode, self.early_stopping.monitor, val))
            torch.save(net, self.model_checkpoint.file_path + '.net.pt')
            torch.save(optim, self.model_checkpoint.file_path + '.optim.pt')
            self.best_not_seen_for = 0
        else:
            self.best_not_seen_for += 1
            if(self.best_not_seen_for == self.early_stopping.patience):
                return True
        return False


FitArgs = namedtuple('FitArgs', ['net', 'training_set', 'validation_set', 'batch_size', 'epochs', 'validation_split', 'callbacks', 'optimizer'])
AdamConfig = namedtuple('AdamConfig', ['lr', 'beta_1', 'beta_2', 'epsilon', 'weight_decay'])
FitReturn = namedtuple('FitReturn', ['history', 'params'])


def get_loader(data_set, batch_size):
    assert isinstance(data_set, datasets.load.DataSet)
    assert isinstance(batch_size, int)

    logger.debug('getting loader for matrices of size:{}'.format(data_set.X.shape[0]))
    assert data_set.X.shape[0] == data_set.y.shape[0] == data_set.w.shape[0] == data_set.z.shape[0]
    logging.debug('loading data with shapes: {} {} {} {}'.format(
        data_set.X.shape, data_set.y.shape, data_set.w.shape, data_set.z.shape
    ))

    x_tensor = torch.LongTensor(data_set.X)

    def reshape_n_n1(x):
        return x.reshape((x.shape[0], 1))

    ywz_data = np.concatenate((reshape_n_n1(data_set.y), reshape_n_n1(data_set.w), reshape_n_n1(data_set.z)), axis=1)
    ywz_tensor = torch.FloatTensor(ywz_data)

    logging.debug('size of x_tensor: {}'.format(x_tensor.shape))
    logging.debug('size of ywz_tensor: {}'.format(ywz_tensor.shape))

    loader = DataLoader(TensorDataset(x_tensor, ywz_tensor), batch_size=batch_size, shuffle=True)

    return loader


def get_metrics(y_predicted_probs, y_true, weights):
    precision, recall = beutil.importance_weighted_precision_recall(
        y_true, y_predicted_probs, weights, 0.5)

    f1 = 2 * precision * recall / (precision + recall + 1e-15)

    precisions, recalls, _ = beutil.importance_weighted_pr_curve(y_true, y_predicted_probs, weights)
    aupr = beutil.area_under_pr_curve(precisions, recalls)

    logging.debug('\npost_training_precision:{}\npost_training_recall:{}'.format(
        precision, recall
    ))

    return (f1, precision, recall, aupr)


def fit(*pargs, **kwargs):
    args = FitArgs(*pargs, **kwargs)
    adam_config = args.optimizer
    logger.debug('fitting with arguments: ' + str(args))

    adam = optim.Adam(args.net.parameters(),
                      lr = adam_config.lr, betas = (adam_config.beta_1, adam_config.beta_2),
                      eps = adam_config.epsilon, weight_decay = adam_config.weight_decay)

    (early_stopping, model_checkpoint, csv_logger) = args.callbacks
    after_epoch = AfterEpoch(early_stopping, model_checkpoint)

    args.net.cuda()

    history = []

    train_loader = get_loader(args.training_set, args.batch_size)
    valid_loader = get_loader(args.validation_set, args.batch_size)

    try:
        for epoch in range(args.epochs):

            u.log_frequently(5, epoch, logger.info, 'starting epoch {} of {}'.format(epoch, args.epochs))

            training_loss = torch.FloatTensor(np.array([0.0])).cuda()
            all_y_true = None
            all_y_prob = None
            all_weights = None

            args.net.train()
            for i, (_X, (_ywz)) in enumerate(train_loader):

                _y, _w, _z = _ywz[:, 0], _ywz[:, 1], _ywz[:, 2]
                logging.debug('shape of _y: {}'.format(_y.shape))
                logging.debug('shape of _w: {}'.format(_w.shape))
                logging.debug('shape of _z: {}'.format(_z.shape))

                # wrap inputs as variables
                X = Variable(_X.cuda())
                y = Variable(_y.cuda())
                # w = Variable(_w.cuda())

                # zero the parameter gradient
                adam.zero_grad()

                # forward
                yp = F.sigmoid(args.net(X))
                yp = yp.resize(yp.size()[0])

                batch_loss = nn.BCELoss(weight=_w.cuda().float())(yp, y.float())

                # metrics
                training_loss += batch_loss.data
                if i == 0:
                    all_y_true = _y.cpu().numpy()
                    all_y_prob = yp.data.cpu().numpy()
                    all_weights = _w.numpy()

                else:  # assuming tensor shape is [batch size, 1] (but this might not be quite right)
                    # logging.info('old: {}, update: {}'.format(all_y_prob.shape, yp.data.cpu().numpy().shape))
                    all_y_true = np.concatenate((all_y_true, _y.cpu().numpy()), axis=0)
                    all_y_prob = np.concatenate((all_y_prob, yp.data.cpu().numpy()), axis=0)
                    all_weights = np.concatenate((all_weights, _w.numpy()), axis=0)


                # backward
                batch_loss.backward()

                # step
                adam.step()

                u.log_frequently(1, i, logger.debug, '{}-th batch training with size {}'.format(i, y.size()))

            training_f1, training_precision, training_recall, training_aupr = get_metrics(
                all_y_prob, all_y_true, all_weights)

            validation_loss = torch.FloatTensor(np.array([0.0])).cuda()
            all_y_true = None
            all_y_prob = None
            all_weights = None

            args.net.eval()
            for i, (_X, _ywz) in enumerate(valid_loader):

                _y, _w, _z = _ywz[:, 0], _ywz[:, 1], _ywz[:, 2]

                X = Variable(_X.cuda())
                y = Variable(_y.cuda())
                # w = Variable(_w.cuda(), requires_grad=False)

                # forward
                yp = F.sigmoid(args.net(X))
                yp = yp.resize(yp.size()[0])
                batch_loss = nn.BCELoss(weight=_w.cuda().float())(yp, y.float())

                # metrics
                validation_loss += batch_loss.data
                if i == 0:
                    all_y_true = _y.cpu().numpy()
                    all_y_prob = yp.data.cpu().numpy()
                    all_weights = _w.numpy()

                else:  # assuming tensor shape is [batch size, 1] (but this might not be quite right)
                    # logging.info('old: {}, update: {}'.format(all_y_prob.shape, yp.data.cpu().numpy().shape))
                    all_y_true = np.concatenate((all_y_true, _y.cpu().numpy()), axis=0)
                    all_y_prob = np.concatenate((all_y_prob, yp.data.cpu().numpy()), axis=0)
                    all_weights = np.concatenate((all_weights, _w.numpy()), axis=0)

                u.log_frequently(10, i, logger.debug, '{}-th batch validating with size {}'.format(i, y.size()))

            validation_f1, validation_precision, validation_recall, validation_aupr = get_metrics(
                all_y_prob, all_y_true, all_weights)

            record = (('loss', training_loss.cpu().numpy()[0]),)\
                     + (('precision', training_precision), ('recall', training_recall))\
                     + (('f1', training_f1), ('aupr', training_aupr))
            record = record\
                     + (('val_loss', validation_loss.cpu().numpy()[0]),)\
                     + (('val_precision', validation_precision), ('val_recall', validation_recall))\
                     + (('val_f1', validation_f1), ('val_aupr', validation_aupr))
            record = OrderedDict(record)
            history.append(record)

            if after_epoch.save_and_stop(record, args.net, adam):
                break

    except KeyboardInterrupt, e:
        logger.exception(e)
        logger.info('interruption in training, continuing')
        pass

    params = {'total_epochs': epoch,
              'es__best_epoch': after_epoch.prev_best_at,
              ('es__best_' + after_epoch.early_stopping.monitor): after_epoch.prev_best}
    history_df = pd.DataFrame.from_records(history)
    history_df.index = history_df.index.rename('epoch')

    return FitReturn(history_df, params)

