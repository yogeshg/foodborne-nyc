from __future__ import print_function
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import logging
logger = logging.getLogger(__name__)

import util

def get_conv_stack(dimensions, filters, kernel_sizes, activation, kernel_l2_regularization, dropout_rate):
    pads = [nn.ConstantPad1d(((i+1)//2, i//2), 0) for i in kernel_sizes]
    convs = [nn.Conv1d(dimensions, filters, i) for i in kernel_sizes]
    # activation, kernel_l2_regularization, dropout_rate not yet used
    return zip(pads, convs)


class Net(nn.Module):

    def __init__(self, maxlen=964, dimensions=300, finetune=False, vocab_size=1000,
            pooling='max', kernel_sizes=(1,2,3), filters=25,
            dropout_rate=0.0, kernel_l2_regularization=0.0,
            lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0,
            embeddings_matrix=None ):
        '''
        maxlen : maximum size of each document
        dimensions : dimension of each vector
        finetune : [True, False] : weather or not to finetune word emdeddings
        vocab_size : size of the vocabulary, emdeddings layer will be this big

        pooling : ['average', 'logsumexp'] : pooling operation for word vectors in a document
        kernel_sizes : tuple : convolve using unigrams / bigrams / trigrams
        filters : None or int : number of filters for convolutional layer
        embeddings_matrix : None or numpy.ndarray : embeddings_matrix to be used for the model

        dropout_rate : float : probability of dropout common accross all the dropout layers
        kernel_l2_regularization : l2 regularization weight
        lr : learning rate for adam optimiser
        beta_1 : parameter for adam optimiser
        beta_2 : parameter for adam optimiser
        epsilon : parameter for adam optimiser
        decay : parameter for adam optimiser
        '''
 
        # Initialize torch model
        super(Net, self).__init__()

        # Validate arguments
        assert (type(maxlen)==int), type(maxlen)
        assert (type(dimensions)==int), type(dimensions)
        assert (type(finetune)==bool), type(finetune)
        assert (type(vocab_size)==int), type(vocab_size)

        assert (pooling in ['max', 'avg', 'logsumexp']), '{} not in {}'.format(str(pooling), str(['max', 'average', 'logsumexp']))
        assert (all(map(isinstance, kernel_sizes, [int]*len(kernel_sizes)))), '{} should all be ints'.format(str(kernel_sizes))
        assert (type(filters)==int), type(filters)
        if isinstance(embeddings_matrix, np.ndarray):
            assert (vocab_size, dimensions) == embeddings_matrix.shape, "mismatched dimensions of embeddings_matrix"
        elif embeddings_matrix is None:
            pass
        else:
            raise TypeError("Unsupported embeddings_matrix type: " + type(embeddings_matrix))

        assert isinstance(dropout_rate, float) 
        assert isinstance(kernel_l2_regularization, float) 
        assert isinstance(lr, float) 
        assert isinstance(beta_1, float) 
        assert isinstance(beta_2, float) 
        assert isinstance(epsilon, float) 
        assert isinstance(decay, float) 

        # save hyperparameters
        self.hyperparameters = {k:v for k,v in locals().iteritems()
                                    if not k in ('embeddings_matrix', 'self')}
    
        logger.info(self.to_json())

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
        for i, (pad, conv) in enumerate(get_conv_stack(dimensions, filters, kernel_sizes, 'relu', kernel_l2_regularization, dropout_rate)):
            setattr(self, 'pad1_' + str(i), pad)
            self.pad1_layers.append('pad1_' + str(i))
            setattr(self, 'conv1_' + str(i), conv)
            self.conv1_layers.append('conv1_' + str(i))

        self.fc = nn.Linear(len(kernel_sizes)*filters, 1)

    def to_json(self, *args, **kwargs):
        kwargs = util.fill_dict(kwargs, {'sort_keys': True, 'indent': 2})
        return json.dumps(self.hyperparameters, *args, **kwargs)

    def forward(self, x):

        logger.debug("activations size: {}".format(x.size()))

        x = self.embeddings(x)
        logger.debug("activations size: {}".format(x.size()))

        def get_padded_conved(x, pad, conv):
            pad1_i = getattr(self, pad)
            conv1_i = getattr(self, conv)
            y = conv1_i(pad1_i(x))
            logger.debug("activations size: {}".format(x.size()))
            return y

        x = x.transpose(1,2)
        x = torch.cat([get_padded_conved(x, pad, conv) for pad, conv in zip(self.pad1_layers, self.conv1_layers)], dim=1)
        logger.debug("activations size: {}".format(x.size()))
        
        x = F.max_pool1d(x, x.size()[-1])
        x = x.transpose(1,2)
        logger.debug("activations size: {}".format(x.size()))

        x = self.fc(x.view(x.size()[0], -1))
        logger.debug("activations size: {}".format(x.size()))
        return (x)

def get_params_list(net, trainable_only=True):
    return [[np.prod(p.size()) for p in l.parameters()] for l in net.children() if (not trainable_only) or l.training]

def get_num_params(net):
    return sum(map(sum, get_params_list(net)))

def test():

    x = Variable(torch.LongTensor(64, 936).random_(1000))
    print(x)

    net = Net()
    print(net)

    print(get_params_list(net, trainable_only=False))
    print(get_params_list(net))
    print(get_num_params(net))

    y = net(x)

def run():
    import main
    main.load_data('twitter.gold', 'data/vocab_yelp.txt', 'data/vectors_yelp.txt', testdata=False)

    (train_size, maxlen) = main.X_train.shape
    (vocab_size, dimensions) = main.embeddings_matrix.shape


    net = Net(embeddings_matrix=main.embeddings_matrix,
                vocab_size=vocab_size, dimensions=dimensions,
                maxlen=maxlen)

    batch_size = 64
    for i in range(train_size//batch_size):
        x = main.X_train[ batch_size*i : batch_size*(i+1) ]
        y = net(Variable(torch.LongTensor(x)))
        logger.info('outpyt: {}'.format(y))
        break

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test()
    pass

