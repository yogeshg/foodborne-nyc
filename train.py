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

from sklearn.model_selection import StratifiedKFold

import datasets.experiments.baseline_experiment_util as beutil
import util as u

DEBUG_TEST_PR_CALCULATION = True

EarlyStopping = namedtuple('EarlyStopping', ['monitor_func', 'patience', 'mode'])
ModelCheckpoint = namedtuple('ModelCheckpoint', ['file_path', 'save_best_only', 'mode'])
CSVLogger = namedtuple('CSVLogger', ['file_path'])

FitArgs = namedtuple('FitArgs', ['net', 'X', 'y', 'w', 'z', 'batch_size', 'epochs', 'validation_split', 'callbacks', 'optimizer'])
AdamConfig = namedtuple('AdamConfig', ['lr', 'beta_1', 'beta_2', 'epsilon', 'decay'])
FitReturn = namedtuple('FitReturn', ['history', 'params'])

class ConfusionMatrix():

    def __init__(self, tp, fp, fn, tn):
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.tn = tn

    def __str__(self):
        return 'ConfusionMatrix(tp={}, fp={}, fn={}, tn={})'.format(self.tp, self.fp, self.fn, self.tn)

    def __add__(self, other):
        assert isinstance(other, ConfusionMatrix)
        return ConfusionMatrix(*map(sum, zip(self, other)))

    def __iadd__(self, other):
        assert isinstance(other, ConfusionMatrix)
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.tn += other.tn
        return self

    def precision(cm, epsilon = 1e-15):
        return (cm.tp / (cm.tp + cm.fp + epsilon))

    def recall(cm, epsilon = 1e-15):
        return (cm.tp / (cm.tp + cm.fn + epsilon))

    def accuracy(cm, epsilon=1e-15):
        return (cm.tp + cm.tn) / (cm.tp + cm.tn + cm.fp + cm.fn + epsilon)

    def f1(cm, epsilon=1e-15):
        a = cm.accuracy()
        p = cm.precision()
        return (2*a*p) / (a+p+epsilon)

    def to_record(self, prefix='', epsilon=1e-15):
        return ((prefix + 'tp', self.tp),
                (prefix + 'fp', self.fp),
                (prefix + 'fn', self.fn),
                (prefix + 'tn', self.tn),
                (prefix + 'precision', self.precision(epsilon=epsilon)),
                (prefix + 'recall', self.recall(epsilon=epsilon)),
                (prefix + 'accuracy', self.accuracy(epsilon=epsilon)),
                (prefix + 'f1', self.f1(epsilon=epsilon)))


def get_loaders(x, y, w, z, start, end, batch_size):
    logger.debug('getting loader for matrices of size {x.shape[0]}, from {start} to {end}'.format(**locals()))
    assert x.shape[0] == y.shape[0] == w.shape[0] == z.shape[0]
    assert 0 <= start and start <= end and end <= x.shape[0]
    assert isinstance(start, int)
    assert isinstance(end, int)

    x_tensor = torch.LongTensor(x[start:end])
    y_tensor = torch.LongTensor(y[start:end])
    w_tensor = torch.FloatTensor(w[start:end])
    z_tensor = torch.LongTensor(z[start:end])

    x_loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size)
    w_loader = DataLoader(TensorDataset(w_tensor, z_tensor), batch_size=batch_size)

    return x_loader, w_loader

# Note: loss is calculated per data point, then weighted by the weight
# vector. This is done based on the suggestions in following issue:
# https://github.com/pytorch/pytorch/issues/264
#

def confusion(y_predicted, y_true, weights):
    assert isinstance(y_predicted, np.ndarray)
    assert isinstance(y_true, np.ndarray)
    assert isinstance(weights, np.ndarray)
    assert y_predicted.shape == y_true.shape == weights.shape

    true_positive = ((y_predicted) & (y_true)) * weights
    false_positive = ((y_predicted) & (~y_true)) * weights
    false_negative = ((~y_predicted) & (y_true)) * weights
    true_negative = ((~y_predicted) & (~y_true)) * weights

    # total_count = true_positive.sum() + false_positive.sum() \
    #               + false_negative.sum() + true_negative.sum()
    # true_positive_count = true_positive.sum()
    # accuracy_numer = true_positive.sum() + true_negative.sum()
    # precision_denom = true_positive.sum() + false_positive.sum()
    # recall_denom = true_positive.sum() + false_negative.sum()
    #
    # logger.debug(str((
    #     ('total_count', total_count),
    #     ('true_positive_count', true_positive_count),
    #     ('accuracy_numer', accuracy_numer),
    #     ('precision_denom', precision_denom),
    #     ('recall_denom', recall_denom)
    # )))

    return ConfusionMatrix(tp=true_positive.sum(), fp=false_positive.sum(),
                           fn=false_negative.sum(), tn=true_negative.sum())


def fit(*pargs, **kwargs):
    args = FitArgs(*pargs, **kwargs)
    adam_config = args.optimizer
    logger.debug('fitting with arguments: ' + str(args))

    # folds = StratifiedKFold(n_splits=n_cv_splits, random_state=1991)

    total_size = args.X.shape[0]
    train_size = int((total_size * (1.00 - args.validation_split)) // 1.0)

    x_train_loader, w_train_loader = get_loaders(args.X, args.y, args.w, args.z, 0, train_size, args.batch_size)
    x_valid_loader, w_valid_loader = get_loaders(args.X, args.y, args.w, args.z, train_size, total_size, args.batch_size)

    entropy_values = nn.CrossEntropyLoss(reduce=False)
    adam = optim.Adam(args.net.parameters(),
                      lr = adam_config.lr, betas = (adam_config.beta_1, adam_config.beta_2),
                      eps = adam_config.epsilon, weight_decay = adam_config.decay)

    def calculate_loss(X, y, yp):
        logits = torch.cat((tt(1 - yp), tt(yp))).t()
        batch_loss_values = entropy_values(logits, y)
        batch_loss = (w.float() * batch_loss_values).sum()
        return batch_loss

    def get_confusion_matrix(yp, _y, _w):
        # Calculate metrics in cpu
        # TODO: can be done in GPU
        y_true = _y.cpu().numpy().astype(bool)
        y_predicted = (yp.data > 0.5).cpu().numpy().flatten().astype(bool)
        weights = _w.cpu().numpy()
        cm = confusion(y_predicted, y_true, weights)
        return cm

    args.net.cuda()

    history = []
    # try:
    for epoch in range(0, args.epochs):

        u.log_frequently(5, epoch, logger.info, 'starting epoch {} of {}'.format(epoch, args.epochs))

        training_loss = torch.FloatTensor(np.array([0.0])).cuda()
        training_cm = ConfusionMatrix(0, 0, 0, 0)

        tt = lambda x: x.view((1, x.size()[0]))

        for i, ((_X, _y), (_w, _z)) in enumerate(zip(x_train_loader, w_train_loader)):

            # wrap inputs as variables
            X = Variable(_X.cuda())
            y = Variable(_y.cuda())
            w = Variable(_w.cuda())

            # zero the parameter gradient
            adam.zero_grad()

            # forward + backward
            # yp = args.net(X)
            # logits = torch.cat((tt(1 - yp), tt(yp))).t()
            # batch_loss_values = entropy_values(logits, y)
            # _batch_loss = (w.float() * batch_loss_values).sum()

            # forward
            yp = args.net(X)
            batch_loss = calculate_loss(X, y, yp)

            # metrics
            cm = get_confusion_matrix(yp, _y, _w)

            training_loss += batch_loss.data
            training_cm += cm

            # backward
            batch_loss.backward()

            # step
            adam.step()

            if(DEBUG_TEST_PR_CALCULATION):
                _precision, _recall = beutil.importance_weighted_precision_recall(
                    _y.cpu().numpy(), yp.data.cpu().numpy().flatten(), _z.numpy(), 0.5)

                assert np.abs(_precision - cm.precision()) < 1e-8, (_precision, cm.precision())
                assert np.abs(_recall - cm.recall()) < 1e-8, (_recall, cm.recall())

                # u.log_frequently(10, i, logger.debug,
                #                  '{}-th batch, precision, recall: {}=={}, {}=={}'.format(
                #                      i, _precision, cm.precision(), _recall, cm.recall()))

            u.log_frequently(10, i, logger.debug, '{}-th batch training with size {}'.format(i, y.size()))

        validation_loss = torch.FloatTensor(np.array([0.0])).cuda()
        validation_cm = ConfusionMatrix(0, 0, 0, 0)

        for i, ((_X, _y), (_w, _)) in enumerate(zip(x_valid_loader, w_valid_loader)):
            X = Variable(_X.cuda())
            y = Variable(_y.cuda())
            w = Variable(_w.cuda())

            # forward
            yp = args.net(X)
            batch_loss = calculate_loss(X, y, yp)

            # metrics
            cm = get_confusion_matrix(yp, _y, _w)

            validation_loss += batch_loss.data
            validation_cm += cm

            u.log_frequently(10, i, logger.debug, '{}-th batch validating with size {}'.format(i, y.size()))

        record = (('loss', training_loss.cpu().numpy()[0]),)\
                 + training_cm.to_record(prefix="")
        record = record\
                 + (('val_loss', validation_loss.cpu().numpy()[0]),)\
                 + validation_cm.to_record(prefix="val_")

        history.append(dict(record))

    # except Exception, e:
    #     logger.exception(e)
    #     logger.info('error in training, continuing')
    #     raise e

    params = {}

    return FitReturn(pd.DataFrame.from_records(history), params)
