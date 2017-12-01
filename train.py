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
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import StratifiedShuffleSplit

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


FitArgs = namedtuple('FitArgs', ['net', 'X', 'y', 'w', 'z', 'batch_size', 'epochs', 'validation_split', 'callbacks', 'optimizer'])
AdamConfig = namedtuple('AdamConfig', ['lr', 'beta_1', 'beta_2', 'epsilon', 'decay'])
FitReturn = namedtuple('FitReturn', ['history', 'params'])


def get_loaders(x, y, w, z, idx, batch_size):
    logger.debug('getting loader for matrices of size:{}, num:{}'.format(x.shape[0], len(idx)))
    assert x.shape[0] == y.shape[0] == w.shape[0] == z.shape[0]

    x_tensor = torch.LongTensor(x[idx])
    y_tensor = torch.FloatTensor(y[idx])
    w_tensor = torch.FloatTensor(w[idx])
    z_tensor = torch.LongTensor(z[idx])

    x_loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size)
    w_loader = DataLoader(TensorDataset(w_tensor, z_tensor), batch_size=batch_size)

    return x_loader, w_loader


def get_metrics(y_predicted_probs, y_true, is_biased):
    precision, recall = beutil.importance_weighted_precision_recall(
        y_true, y_predicted_probs, is_biased.astype(bool), 0.5)

    f1 = 2 * precision * recall / (precision + recall + 1e-15)

    precisions, recalls, _ = beutil.importance_weighted_pr_curve(y_true, y_predicted_probs, is_biased.astype(bool))
    aupr = beutil.area_under_pr_curve(precisions, recalls)

    logging.debug('\npost_training_precision:{}\npost_training_recall:{}'.format(
        precision, recall
    ))

    return (f1, precision, recall, aupr)


def fit(*pargs, **kwargs):
    args = FitArgs(*pargs, **kwargs)
    adam_config = args.optimizer
    logger.debug('fitting with arguments: ' + str(args))

    entropy_values = nn.CrossEntropyLoss(reduce=False)
    adam = optim.Adam(args.net.parameters(),
                      lr = adam_config.lr, betas = (adam_config.beta_1, adam_config.beta_2),
                      eps = adam_config.epsilon, weight_decay = adam_config.decay)

    (early_stopping, model_checkpoint, csv_logger) = args.callbacks
    after_epoch = AfterEpoch(early_stopping, model_checkpoint)

    args.net.cuda()

    history = []
    folds = StratifiedShuffleSplit(n_splits=args.epochs, test_size=args.validation_split, random_state=1991)
    label_bias_tuples = ['{},{}'.format(y,b) for y,b in zip(args.y, args.z)]

    try:
        for epoch, (training_idx, validation_idx) in enumerate(folds.split(np.zeros(len(args.z)), label_bias_tuples)):
            x_train_loader, w_train_loader = get_loaders(args.X, args.y, args.w, args.z, training_idx, args.batch_size)
            x_valid_loader, w_valid_loader = get_loaders(args.X, args.y, args.w, args.z, validation_idx, args.batch_size)

            u.log_frequently(5, epoch, logger.info, 'starting epoch {} of {}'.format(epoch, args.epochs))

            training_loss = torch.FloatTensor(np.array([0.0])).cuda()
            all_y_true = None
            all_y_prob = None
            all_is_biased = None

            for i, ((_X, _y), (_w, _z)) in enumerate(zip(x_train_loader, w_train_loader)):

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
                    all_is_biased = _z.numpy()

                else:  # assuming tensor shape is [batch size, 1] (but this might not be quite right)
                    # logging.info('old: {}, update: {}'.format(all_y_prob.shape, yp.data.cpu().numpy().shape))
                    all_y_true = np.concatenate((all_y_true, _y.cpu().numpy()), axis=0)
                    all_y_prob = np.concatenate((all_y_prob, yp.data.cpu().numpy()), axis=0)
                    all_is_biased = np.concatenate((all_is_biased, _z.numpy()), axis=0)


                # backward
                batch_loss.backward()

                # step
                adam.step()

                u.log_frequently(1, i, logger.debug, '{}-th batch training with size {}'.format(i, y.size()))

            training_f1, training_precision, training_recall, training_aupr = get_metrics(
                all_y_prob, all_y_true, all_is_biased)

            validation_loss = torch.FloatTensor(np.array([0.0])).cuda()
            all_y_true = None
            all_y_prob = None
            all_is_biased = None

            for i, ((_X, _y), (_w, _z)) in enumerate(zip(x_valid_loader, w_valid_loader)):
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
                    all_is_biased = _z.numpy()

                else:  # assuming tensor shape is [batch size, 1] (but this might not be quite right)
                    # logging.info('old: {}, update: {}'.format(all_y_prob.shape, yp.data.cpu().numpy().shape))
                    all_y_true = np.concatenate((all_y_true, _y.cpu().numpy()), axis=0)
                    all_y_prob = np.concatenate((all_y_prob, yp.data.cpu().numpy()), axis=0)
                    all_is_biased = np.concatenate((all_is_biased, _z.numpy()), axis=0)

                u.log_frequently(10, i, logger.debug, '{}-th batch validating with size {}'.format(i, y.size()))

            validation_f1, validation_precision, validation_recall, validation_aupr = get_metrics(
                all_y_prob, all_y_true, all_is_biased)

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

    params = {'epoch': epoch,
              'es__best_at': after_epoch.prev_best_at, 'es__best_val': after_epoch.prev_best}
    history_df = pd.DataFrame.from_records(history)
    history_df.index = history_df.index.rename('epoch')

    return FitReturn(history_df, params)

