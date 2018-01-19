from __future__ import print_function

import json
import argparse
from collections import Counter
import logging
logger = logging.getLogger(__name__)

import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.utils.extmath import softmax

import models
import datasets.load
import util as u

class Inspector:

    def __init__(self, net, required_embeddings):
        try:
            assert isinstance(net, nn.Module)
            assert isinstance(required_embeddings, np.ndarray)
        except Exception as e:
            logger.exception(e)
        self.net = net
        # self.created_indexer = created_indexer
        # self.required_indexer = required_indexer
        vocab_size, dimensions = required_embeddings.shape
        self.net.embeddings = nn.Embedding(vocab_size, dimensions)
        self.net.embeddings.weight.data.copy_(torch.FloatTensor(required_embeddings))
        self.net.cuda()

    def predict_proba(self, x, batch_size=64):
        # x = np.array(x)
        # # print((x.shape[0], 1))
        # x = x.reshape((-1, 1))
        logger.debug('predict_proba on vector of type {} and shape {}'.format(type(x), x.shape))

        all_logits = []

        for batch_id in range(0, np.ceil(x.shape[0] / float(batch_size)).astype(int)):
            batch_start = batch_size * batch_id
            batch_end = batch_size * (batch_id + 1)

            X0 = Variable(torch.cuda.LongTensor(x[batch_start:batch_end]))
            logits_torch = self.net.forward(X0)
            logits = logits_torch.data.cpu().numpy()
            all_logits.extend(np.concatenate((1 - logits, logits), axis=1))
            u.log_frequently(100, batch_id, logger.debug, 'processed batch, shape of logits: {}'.format(logits.shape))
        class_probs = softmax(np.array(all_logits))
        logger.debug('shape of class_probs: {}'.format(class_probs.shape))
        return class_probs

    def predict(self, x, batch_size=64):
        logger.debug('predict on vector of type {}'.format(type(x)))
        class_probs = self.predict_proba(x, batch_size=batch_size)
        predictions = np.argmax(class_probs, axis=1)
        logger.debug('shape of predictions: {}'.format(predictions.shape))
        return predictions

