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
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.utils.extmath import softmax

from models import log_sum_exp, equal_pad
import datasets.load
import util as u

class Inspector:

    def __init__(self, net, required_embeddings):
        try:
            assert isinstance(net, nn.Module)
            assert isinstance(required_embeddings, np.ndarray)
        except Exception as e:
            logger.exception(e)
        logger.info('initialising inspector for net with hyperparams: {}'.format(net.to_json()))
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

    def forward_inspect(self, X0, indexer):
        # logger.debug("activations size: {}".format(x.size()))

        # X0.shape
        # torch.Size([64, 36])
        X0_data = X0.cpu().data.numpy()

        # 1. Apply Embeddings
        X1 = self.net.embeddings(X0)

        # X1.shape
        # torch.Size([64, 36, 200])
        activations_embeddings = X1.norm(dim=2).cpu().data.numpy()
        argmax_embeddings = activations_embeddings.argmax(axis=1)

        # logger.debug("activations size: {}".format(x.size()))

        # 2. Apply Convolutions
        def compose_stack(x, pad, conv, drop):
            pad1_i = getattr(self.net, pad)
            conv1_i = getattr(self.net, conv)
            drop1_i = getattr(self.net, drop)
            y = drop1_i(conv1_i(pad1_i(x)))
            # logger.debug("activations size: {}".format(x.size()))
            return y

        stack_layer_names = zip(self.net.pad1_layers, self.net.conv1_layers, self.net.drop1_layers)
        stack_layers = [compose_stack(X1.transpose(1, 2), pad, conv, drop) for pad, conv, drop in stack_layer_names]
        X2 = torch.cat(stack_layers, dim=1)
        # logger.debug("activations size: {}".format(x.size()))

        # X2.shape
        # torch.Size([64, 60, 37])
        activations_conv = X2.cpu().data.numpy()
        ngrams_heatmaps = []
        ngrams_interest = []
        for i, activations_conv_i in enumerate(activations_conv):
            ngrams_interest.append([])
            ngrams_map = Counter()
            for j, activations_conv_i_j in enumerate(activations_conv_i):
                # activation for i-th data point, j-th convolutional filter
                jj = j // self.net.hyperparameters['filters']
                kernel_size = self.net.hyperparameters['kernel_sizes'][jj]
                pre, post = equal_pad(kernel_size)
                sentence = X0_data[i]
                argmax_conv_i_j = activations_conv_i_j.argmax()
                interest = activations_conv_i_j.max()
                ngram_location = slice(max(0, argmax_conv_i_j - pre), min(len(sentence), argmax_conv_i_j + post))
                ngram = sentence[ngram_location]
                ngrams_interest[-1].append((ngram_location, interest, ngram))
                ngrams_map[tuple(ngram)] += interest
            ngrams_heatmap = [("_".join(map(indexer.get_token, k)), v) for k, v in ngrams_map.items()]
            # print(ngrams_heatmap)
            # print(" ".join(map(indexer.get_token, sentence)))
            # for k, v in (sorted(ngrams_heatmap, key=lambda x: x[1])):
            #     print("{}:\t{}".format(v, k))
            ngrams_heatmaps.append(ngrams_heatmap)

        # 3. Apply pooling, activations
        if (self.net.conv1_stack_pooling == 'max'):
            X31 = F.max_pool1d(X2, X2.size()[-1])
            # logger.debug("activations size: {}".format(x.size()))
            X3 = X31.transpose(1, 2)
        elif (self.net.conv1_stack_pooling == 'average'):
            X31 = F.avg_pool1d(X2, X2.size()[-1])
            # logger.debug("activations size: {}".format(x.size()))
            X3 = X31.transpose(1, 2)
        elif (self.net.conv1_stack_pooling == 'logsumexp'):
            X31 = log_sum_exp(X2, dim=2, keepdim=True)
            # logger.debug("activations size: {}".format(x.size()))
            X3 = X31.transpose(1, 2)
        else:
            raise RuntimeError, 'Unexpected pooling', self.net.conv1_stack_pooling

        if (self.net.conv1_stack_activation == 'relu'):
            X4 = F.relu(X3)
        else:
            raise RuntimeError, 'Unexpected activation', self.net.conv1_stack_activation

        # logger.debug("activations size: {}".format(x.size()))

        # 4. Apply fully connected layer
        X5 = self.net.fc(X4.view(X4.size()[0], -1))
        # logger.debug("activations size: {}".format(x.size()))

        (weights, bias) = map(lambda x: x.cpu().data.numpy(), self.net.fc.parameters())
        # print(weights, bias)

        return X5, weights, bias, ngrams_interest

def get_heatmap(idx, weights, ngrams_interest):
    heatmap_pos = Counter()
    heatmap_neg = Counter()
    for w, (location, a, indices) in zip(weights[0], ngrams_interest[idx]):
        p = w * a
        location = np.array(range(1000))[location]
        ngram_len = len(location)
        for l in location:
            if p > 0:
                heatmap_pos[l] += float(p) / ngram_len
            else:
                heatmap_neg[l] += float(p) / ngram_len

    return (heatmap_pos, heatmap_neg)

