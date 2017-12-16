from __future__ import print_function
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import math
from collections import Counter

import numpy as np
import logging

logger = logging.getLogger(__name__)

import util

def equal_pad(kernel_size):
    return ((kernel_size + 1) // 2, kernel_size // 2)

def get_conv_stack(dimensions, filters, kernel_sizes, dropout_rate):
    pads = [nn.ConstantPad1d(equal_pad(i), 0) for i in kernel_sizes]
    convs = [nn.Conv1d(dimensions, filters, i) for i in kernel_sizes]
    drops = [nn.Dropout(p=dropout_rate) for i in kernel_sizes]
    return zip(pads, convs, drops)

def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)

class Net(nn.Module):
    def __init__(self, dimensions=200, finetune=False, vocab_size=1000,
                 pooling='max', activation='relu', kernel_sizes=(1, 2, 3), filters=5, dropout_rate=0.0,
                 lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, weight_decay=0.0,
                 embeddings_matrix=None):
        """

        :param dimensions: int: dimension of each vector
        :param finetune: bool : weather or not to finetune word emdeddings
        :param vocab_size: int: size of the vocabulary, emdeddings layer will be this big

        :param pooling: ['average', 'logsumexp']: pooling operation for word vectors in a document
        :param activation: str: activation for convolutional stack
        :param kernel_sizes: tuple: convolve using unigrams / bigrams / trigrams
        :param filters: int : number of filters for convolutional layer
        :param dropout_rate: float: probability of dropout common across all the dropout layers

        :param lr: learning rate for adam optimiser
        :param beta_1: parameter for adam optimiser
        :param beta_2: parameter for adam optimiser
        :param epsilon: parameter for adam optimiser
        :param weight_decay: parameter for adam optimiser (l2 regularization weight, kernel_l2_regularization)

        :param embeddings_matrix: None or numpy.ndarray : embeddings_matrix to be used for the model
        """

        # Initialize torch model
        super(Net, self).__init__()

        # Validate arguments
        assert (type(dimensions)==int), type(dimensions)
        assert (type(finetune)==bool), type(finetune)
        assert (type(vocab_size)==int), type(vocab_size)

        util.assert_in(pooling, ['max', 'average', 'logsumexp'])
        assert (all(map(lambda x: isinstance(x, int), kernel_sizes))), '{} should all be ints'.format(str(kernel_sizes))
        assert (isinstance(filters,int)), type(filters)
        assert isinstance(dropout_rate, float)

        assert isinstance(lr, float)
        assert isinstance(beta_1, float)
        assert isinstance(beta_2, float)
        assert isinstance(epsilon, float)
        assert isinstance(weight_decay, float)

        if isinstance(embeddings_matrix, np.ndarray):
            assert (vocab_size, dimensions) == embeddings_matrix.shape, "mismatched dimensions of embeddings_matrix"
        elif embeddings_matrix is None:
            pass
        else:
            raise TypeError("Unsupported embeddings_matrix type: " + type(embeddings_matrix))

        # save hyperparameters
        self.hyperparameters = {k: v for k, v in locals().iteritems()
                                if not k in ('embeddings_matrix', 'self')}
        logger.debug(self.to_json(indent=None))

        # our layers 

        # Pass the input through embeddings
        # https://discuss.pytorch.org/t/can-we-use-pre-trained-word-embeddings-for-weight-initialization-in-nn-embedding/1222/12
        self.embeddings = nn.Embedding(vocab_size, dimensions)
        self.embeddings.training = finetune
        if not embeddings_matrix is None:
            self.embeddings.weight.data.copy_(torch.FloatTensor(embeddings_matrix))

        # add droupout layer
        # self.dropout = nn.Dropout(p=dropout_rate)

        # get the convolutional stack
        self.pad1_layers = []
        self.conv1_layers = []
        self.drop1_layers = []
        conv_stack = get_conv_stack(dimensions, filters, kernel_sizes, dropout_rate)

        for i, (pad, conv, drop) in enumerate(conv_stack):
            setattr(self, 'pad1_' + str(i), pad)
            self.pad1_layers.append('pad1_' + str(i))
            setattr(self, 'conv1_' + str(i), conv)
            self.conv1_layers.append('conv1_' + str(i))
            setattr(self, 'drop1_' + str(i), drop)
            self.drop1_layers.append('drop1_' + str(i))

        self.conv1_stack_pooling = pooling
        self.conv1_stack_activation = activation

        self.fc = nn.Linear(len(kernel_sizes) * filters, 1)

    def to_json(self, *args, **kwargs):
        kwargs = util.fill_dict(kwargs, {'sort_keys': True, 'indent': 2})
        return json.dumps(self.hyperparameters, *args, **kwargs)

    def __str__(self):
        s1 = super(Net, self).__str__()
        s2 = "all_parameters: {}".format(get_params_list(self, trainable_only=False))
        s3 = "trainable_parameters: {}".format(get_num_params(self))
        return "\n".join((s1, s2, s3))

    def forward(self, x):

        # logger.debug("activations size: {}".format(x.size()))

        # 1. Apply Embeddings
        x = self.embeddings(x)
        # logger.debug("activations size: {}".format(x.size()))

        # 2. Apply Convolutions
        def compose_stack(x, pad, conv, drop):
            pad1_i = getattr(self, pad)
            conv1_i = getattr(self, conv)
            drop1_i = getattr(self, drop)
            y = drop1_i(conv1_i(pad1_i(x)))
            # logger.debug("activations size: {}".format(x.size()))
            return y

        x = x.transpose(1, 2)
        stack_layer_names = zip(self.pad1_layers, self.conv1_layers, self.drop1_layers)
        stack_layers = [compose_stack(x, pad, conv, drop) for pad, conv, drop in stack_layer_names]
        x = torch.cat(stack_layers, dim=1)
        # logger.debug("activations size: {}".format(x.size()))

        # 3. Apply pooling, activations
        if(self.conv1_stack_pooling=='max'):
            x = F.max_pool1d(x, x.size()[-1])
            # logger.debug("activations size: {}".format(x.size()))
            x = x.transpose(1, 2)
        elif (self.conv1_stack_pooling == 'average'):
            x = F.avg_pool1d(x, x.size()[-1])
            # logger.debug("activations size: {}".format(x.size()))
            x = x.transpose(1, 2)
        elif (self.conv1_stack_pooling == 'logsumexp'):
            x = log_sum_exp(x, dim=2, keepdim=True)
            # logger.debug("activations size: {}".format(x.size()))
            x = x.transpose(1, 2)
        else:
            raise RuntimeError, 'Unexpected pooling', self.conv1_stack_pooling

        if(self.conv1_stack_activation=='relu'):
            x = F.relu(x)
        else:
            raise RuntimeError, 'Unexpected activation', self.conv1_stack_activation

        # logger.debug("activations size: {}".format(x.size()))

        # 4. Apply fully connected layer
        x = self.fc(x.view(x.size()[0], -1))
        # logger.debug("activations size: {}".format(x.size()))
        return x


def forward_inspect(self, X0, indexer):
    # logger.debug("activations size: {}".format(x.size()))

    # X0.shape
    # torch.Size([64, 36])
    X0_data = X0.cpu().data.numpy()

    # 1. Apply Embeddings
    X1 = self.embeddings(X0)

    # X1.shape
    # torch.Size([64, 36, 200])
    activations_embeddings= X1.norm(dim=2).cpu().data.numpy()
    argmax_embeddings = activations_embeddings.argmax(axis=1)

    # logger.debug("activations size: {}".format(x.size()))

    # 2. Apply Convolutions
    def compose_stack(x, pad, conv, drop):
        pad1_i = getattr(self, pad)
        conv1_i = getattr(self, conv)
        drop1_i = getattr(self, drop)
        y = drop1_i(conv1_i(pad1_i(x)))
        # logger.debug("activations size: {}".format(x.size()))
        return y

    stack_layer_names = zip(self.pad1_layers, self.conv1_layers, self.drop1_layers)
    stack_layers = [compose_stack(X1.transpose(1, 2), pad, conv, drop) for pad, conv, drop in stack_layer_names]
    X2 = torch.cat(stack_layers, dim=1)
    # logger.debug("activations size: {}".format(x.size()))

    # X2.shape
    # torch.Size([64, 60, 37])
    activations_conv = X2.cpu().data.numpy()
    for i, activations_conv_i in enumerate(activations_conv):
        ngrams_interest = []
        ngrams_map = Counter()
        for j, activations_conv_i_j in enumerate(activations_conv_i):
            # activation for i-th data point, j-th convolutional filter
            jj = j // self.hyperparameters['filters']
            kernel_size = self.hyperparameters['kernel_sizes'][jj]
            pre, post = equal_pad(kernel_size)
            sentence = X0_data[i]
            argmax_conv_i_j = activations_conv_i_j.argmax()
            interest = activations_conv_i_j.max()
            ngram_location = slice(max(0, argmax_conv_i_j - pre), min(len(sentence), argmax_conv_i_j + post))
            ngram = sentence[ngram_location]
            ngrams_interest.append((ngram_location, interest, ngram))
            ngrams_map[tuple(ngram)] += interest
        print(json.dumps({indexer.get_token(k):v for k, v in ngrams_map.items()}, indent=0))

    # 3. Apply pooling, activations
    if (self.conv1_stack_pooling == 'max'):
        X31 = F.max_pool1d(X2, X2.size()[-1])
        # logger.debug("activations size: {}".format(x.size()))
        X3 = X31.transpose(1, 2)
    elif (self.conv1_stack_pooling == 'average'):
        X31 = F.avg_pool1d(X2, X2.size()[-1])
        # logger.debug("activations size: {}".format(x.size()))
        X3 = X31.transpose(1, 2)
    elif (self.conv1_stack_pooling == 'logsumexp'):
        X31 = log_sum_exp(X2, dim=2, keepdim=True)
        # logger.debug("activations size: {}".format(x.size()))
        X3 = X31.transpose(1, 2)
    else:
        raise RuntimeError, 'Unexpected pooling', self.conv1_stack_pooling

    if (self.conv1_stack_activation == 'relu'):
        X4 = F.relu(X3)
    else:
        raise RuntimeError, 'Unexpected activation', self.conv1_stack_activation

    # logger.debug("activations size: {}".format(x.size()))

    # 4. Apply fully connected layer
    X5 = self.fc(X4.view(X4.size()[0], -1))
    # logger.debug("activations size: {}".format(x.size()))
    return X5


def get_params_list(net, trainable_only=True):
    return [[np.prod(p.size()) for p in l.parameters()] for l in net.children() if (not trainable_only) or l.training]


def get_num_params(net):
    return sum(map(sum, get_params_list(net)))


def test():
    x = Variable(torch.LongTensor(64, 936).random_(1000))
    print(x)

    for pooling in ('max', 'average', 'logsumexp'):
        net = Net(pooling=pooling)
        print(net)

        print(get_params_list(net, trainable_only=False))
        print(get_params_list(net))
        print(get_num_params(net))

        y = net(x)


def run():
    import main
    main.load_data('twitter.gold', 'data/vocab_yelp.txt', 'data/vectors_yelp.txt')

    (train_size, _) = main.X_train.shape
    (vocab_size, dimensions) = main.embeddings_matrix.shape

    for pooling in ('max', 'average', 'logsumexp'):
        net = Net(pooling=pooling, embeddings_matrix=main.embeddings_matrix,
              vocab_size=vocab_size, dimensions=dimensions)

        batch_size = 64
        for i in range(train_size // batch_size):
            x = main.X_train[batch_size * i: batch_size * (i + 1)]
            y = net(Variable(torch.LongTensor(x)))
            logger.info('output: {}'.format(y))
            break


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test()
    pass
