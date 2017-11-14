from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np

def get_conv_stack(dimensions, filters, kernel_sizes, activation, kernel_l2_regularization, dropout_rate):
    pads = [nn.ConstantPad1d(((i+1)//2, i//2), 0) for i in kernel_sizes]
    convs = [nn.Conv1d(dimensions, filters, i) for i in kernel_sizes]
    # activation, kernel_l2_regularization, dropout_rate not yet used
    return zip(pads, convs)


class Net(nn.Module):

    def __init__(self, maxlen=964, dimensions=300, finetune=False, vocab_size=1000,
            pooling='max', kernel_sizes=(1,2,3), filters=50, weights=None,
            dropout_rate=0, kernel_l2_regularization=0,
            lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0 ):

        super(Net, self).__init__()
       
        # our layers 
        
        # Pass the input through embeddings
        # https://discuss.pytorch.org/t/can-we-use-pre-trained-word-embeddings-for-weight-initialization-in-nn-embedding/1222/12
        self.embeddings = nn.Embedding(vocab_size, dimensions)
        self.embeddings.training = finetune

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

    def forward(self, x):

        print(x.size())

        x = self.embeddings(x)
        print(x.size())

        def get_padded_conved(x, pad, conv):
            pad1_i = getattr(self, pad)
            conv1_i = getattr(self, conv)
            y = conv1_i(pad1_i(x))
            print(y.size())
            return y

        x = x.transpose(1,2)
        x = torch.cat([get_padded_conved(x, pad, conv) for pad, conv in zip(self.pad1_layers, self.conv1_layers)], dim=1)
        print(x.size())
        
        x = F.max_pool1d(x, 965)
        x = x.transpose(1,2)
        print(x.size())

        x = self.fc(x.view(x.size()[0], -1))
        print(x.size())
        return (x)

def get_trainable_params_list(net):
    return [[np.prod(p.size()) for p in l.parameters()] for l in net.children() if l.training]

def get_trainable_params_number(net):
    return sum(map(sum, get_trainable_params_list(net)))

def test():

    x = Variable(torch.LongTensor(64, 964).random_(1000))
    print(x)

    net = Net()
    print(net)

    print(get_trainable_params_list(net))
    print(get_trainable_params_number(net))

    y = net(x)

if __name__ == '__main__':
    test()



