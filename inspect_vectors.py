import torch
from torch.autograd import Variable

import models
import main
import util

import os
from collections import Counter
from itertools import product
from contextlib import contextmanager

import numpy as np
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)

from collections import defaultdict

import cPickle

selected_models = {
"twitter.gold"   : "data/models/20171217_020721_811949/",
"twitter.silver" : "data/models/20171217_175028_811949/",
"twitter.biased" : "data/models/20171217_022127_811949/",
"yelp.gold"      : "data/models/20171217_061943_811949/",
"yelp.silver"    : "data/models/20171217_195647_811949/",
"yelp.biased"    : "data/models/20171217_203244_811949/",
}


dataset_media = ('twitter', 'yelp')
dataset_regimes = ('gold',)

data_paths = ('data/twitter_data/', 'data/yelp_data/')
embeddings_paths = ('data/glove.twitter.27B.200d.txt', 'data/glove.840B.300d.txt')

inputs = list(product(zip(dataset_media, data_paths, embeddings_paths), dataset_regimes))


def get_ngrams(net):
    for layer_name in net.conv1_layers:
        layer = getattr(net, layer_name)

        weights, biases = layer.parameters()
        for ngram, bias in zip(weights, biases):
            yield ngram.data.numpy(), bias.data.numpy(), layer_name

def ngram_to_vectors(ngram):
    n = ngram.shape[1]
    return [ngram[:,i] for i in range(n)]


class BiggestRecorder:

    def __init__(self, num):
        self.biggest_values = [float('-inf') for _ in range(num)]
        self.biggest_keys = [None for _ in range(num)]
        self.num = num

    def check_and_record(self, key, value):
        index = np.searchsorted(self.biggest_values, value)
        if(index > 0):
            self.biggest_values.insert(index, value)
            self.biggest_values.pop(0)
            self.biggest_keys.insert(index, key)
            self.biggest_keys.pop(0)
            assert self.num == len(self.biggest_values)
            assert self.num == len(self.biggest_keys)

    def union(self, br):
        assert isinstance(br, BiggestRecorder)
        for key, value in zip(br.biggest_values, br.biggest_keys):
            self.check_and_record(key, value)

    def __repr__(self):
        return """BiggestRecorder(num = {},
        biggest_values = {},
        biggest_keys = {},
        )
        """.format(self.num, self.biggest_values, self.biggest_keys)

# def reduce_biggest_recorders(br1, br2):
#     br = BiggestRecorder(min(br1.num, br2.num))
#     for key, value in zip(br1.biggest_keys + br2.biggest_keys, br1.biggest_values + br2.biggest_values):
#         br.check_and_record(key, value)
#     return br

class GloveVectors:

    def __init__(self, fname, dimensions='infer', head=10):
        self.fname = fname
        self.dimensions = dimensions
        self.head = head
        # self.tokens = []
        # self.vectors = None
        # self.indices = {}
        self.size = 0

    def parse_lines(self, f):
        for i, line in enumerate(f):
            if i >= self.head:
                break
            util.log_frequently(100000, i, logger.info, "vectors visited")
            words = line.split()
            token = words[0]
            vector = map(float, words[1:])
            yield (token, vector)

    def reload(self):
        tokens = list()
        with open(self.fname, 'r') as f:
            for token, vector in self.parse_lines(f):
                tokens.append(token)
                # vectors.append(vector)
        # self.tokens = tokens
        # self.vectors = np.array(vectors)
        # self.indices = dict((x,i) for i,x in enumerate(self.tokens))
        self.size = len(tokens)

    def get_biggest_vectors(self, vectors, num, start=0, end=float('inf')):
        brs = [BiggestRecorder(num) for _ in range(len(vectors))]
        with open(self.fname, 'r') as f:
            for j, (token, v) in enumerate(self.parse_lines(f)):
                #util.log_frequently(100000, j, logger.info, "{}".format(brs))
                if j >= start and j < end:
                    for i,u in enumerate(vectors):
                        d = np.dot(u, v)
                        brs[i].check_and_record((token, v), d)
        return brs


def raze(l):
    for i in l:
        for j in i:
            yield j


for (medium, data_path, embeddings_path), regime in inputs:
    dataset = medium + '.' + regime
    model_dirname = selected_models[dataset]

    logger.info("loading model")
    net = torch.load(os.path.join(model_dirname, 'checkpoint.net.pt'), map_location=lambda storage, y: storage)
    ngrams = list(get_ngrams(net))
    vectors = map(ngram_to_vectors, map(lambda x:x[0], ngrams))
    all_vectors = list(raze(vectors))

    logger.info("loading vectors")
    gv = GloveVectors(embeddings_path, head=float('inf'))
    gv.reload()

    logger.info("getting closest vectors")
    closest_vectors = gv.get_biggest_vectors(all_vectors, 100)
    # closest_vectors_tokens = map(lambda i: gv.tokens[i], closest_vectors[0])
    # closest_vectors_all = closest_vectors + (closest_vectors_tokens,)

    logger.info("saving results")
    with open(medium+'.closesetvectors.pkl', 'w') as f:
        cPickle.dump((
            (ngrams, all_vectors) + closest_vectors
        ), f)


