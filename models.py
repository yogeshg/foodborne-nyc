import sys
import numpy as np
import foodbornenyc.util.util as u

import logging
logging.basicConfig(level = logging.INFO, format=
        '%(asctime)s:%(levelname)s:%(name)s:%(threadName)s:line %(lineno)d: %(message)s')

from keras.preprocessing import sequence
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Embedding, GlobalMaxPooling1D, GlobalAveragePooling1D, Convolution1D, merge, Dropout
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.utils
from keras.optimizers import Adam
from keras.engine.topology import Layer
from keras import backend as K
import tensorflow as tf
from datasets import yelp
from metrics import auc

from util.archiver import get_archiver

import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = u.get_logger(__name__)

import config as c

def add_defaults(d1, d2):
     d22 = dict(d2)
     d11 = dict(d1)
     d22.update(d11)
     d11.update(d22)
     return d11

class LogSumExpPooling(Layer):

    def call(self, x):
        # could be axis 0 or 1
        return tf.reduce_logsumexp(x, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[:1]+input_shape[2:]

def get_conv_stack(input_layer, nb_filter, filter_lengths, activation, kernel_l2_regularization):
    layers = [Convolution1D(nb_filter=nb_filter, filter_length=f,
            border_mode='same', activation=activation, kernel_regularizer=regularizers.l2(kernel_l2_regularization),
            subsample_length=1)(input_layer) for f in filter_lengths]
    if (len(layers) <= 0):
        return input_layer
    elif (len(layers) == 1):
        return layers[0]
    else:
        return merge(layers, mode='concat')

def get_model(maxlen=964, dimensions=200, finetune=False, vocab_size=1000,
            pooling='max', filter_lengths=(), nb_filter=0, weights=None,
            dropout_rate=0, kernel_l2_regularization=0,
            lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0 ):
    '''
    maxlen : maximum size of each document
    dimensions : dimension of each vector
    finetune : [True, False] : weather or not to finetune word emdeddings
    vocab_size : size of the vocabulary, emdeddings layer will be this big
    pooling : ['average', 'logsumexp'] : pooling operation for word vectors in a document
    filter_lengths : tuple : convolve using unigrams / bigrams / trigrams
    nb_filter : None or int : number of filters for convolutional layer
    '''
    assert(type(dimensions)==int), type(dimensions)
    assert(type(maxlen)==int), type(maxlen)
    assert(type(finetune)==bool), type(finetune)
    assert(type(vocab_size)==int), type(vocab_size)
    assert(pooling in ['max', 'avg', 'logsumexp']), '{} not in {}'.format(str(pooling), str(['max', 'average', 'logsumexp']))
    assert (all([x in (1,2,3) for x in filter_lengths])), '{} not in {}'.format(str(filter_lengths), str((1,2,3)))
    assert (type(nb_filter)==int), type(nb_filter)
    params = {k:v for k,v in locals().iteritems() if k!='weights'}

    print params # TODO print into file

    doc_input = Input(shape=(maxlen,), dtype='int32')
    embedding_layer = Embedding(vocab_size, dimensions, weights=weights, input_length=maxlen, trainable=finetune)
    y = embedding_layer(doc_input)
    y = Dropout(dropout_rate, noise_shape=None, seed=None)(y)
    y = get_conv_stack(y, nb_filter, filter_lengths, 'relu', kernel_l2_regularization)
    if(pooling=='max'):
        y = GlobalMaxPooling1D()(y)
    elif(pooling=='avg'):
        y = GlobalAveragePooling1D()(y)
    elif(pooling=='logsumexp'):
        y = LogSumExpPooling()(y)
    else:
        assert(pooling in ['max', 'logsumexp']), '{} not implemented yet'.format(pooling)

    y = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(kernel_l2_regularization))(y)

    model = Model(doc_input, y)
    model.summary() # TODO print into file
    adam = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy', auc])

    return (model, params)

def plot_metric(df, metric_name, i, dirpath):
    assert type(df) == pd.DataFrame, type(df)
    assert type(metric_name) == str, type(metric_name)
    assert type(i) == int, i
    assert type(dirpath) == str, dirpath
    val_metric = 'val_{}'.format(metric_name)
    cname = 'val_{}_{:04d}'.format(metric_name, i)
    df.loc[:, cname] = df.loc[i, val_metric]
    df.loc[:, [metric_name, val_metric, cname]].plot()
    plt.savefig(dirpath + '/{}.png'.format(metric_name))
    return

def plot_model(*args, **kwargs):
    output = None
    try:
        output = keras.utils.plot_model(*args, **kwargs)
    except Exception as e:
        logger.exception(e)
    return output

def save_history(history, dirpath):
    with open(dirpath+'/training.json', 'w') as f:
        json.dump(history.params, f, indent=2)

    df = pd.DataFrame.from_dict(history.history)
    df.to_csv(dirpath+'/history.csv')
    i = df.loc[:, c.monitor].argmax()

    for m in c.metrics + ['loss']:
        plot_metric(df, m, i, dirpath)

    return

def load_data(datapath, indexpath, embeddingspath, testdata=False):
    global embeddings_matrix, X_train, y_train, X_test, y_test
    if( testdata ):
        datapath = '/tmp/yo/foodborne/yelp_labelled_sample.csv'
        indexpath = '/tmp/yo/foodborne/vocab_yelp_sample.txt'
        embeddingspath = '/tmp/yo/foodborne/vectors_yelp_sample.txt'
    
    embeddings_matrix = yelp.load_embeddings_matrix(indexpath, embeddingspath)
    ((X_train, y_train), (X_test, y_test), _) = yelp.load_devset_testset_index(datapath, indexpath)
    # ratio_train_validate = 0.8
    # # TODO remove this ration from here and add directly to model.fit
    # ((X_train, y_train), (X_validate, y_validate)) = yelp.cutXY((X, y), ratio_train_validate)

    for x in (X_train, y_train, X_test, y_test):
        logger.debug("shape and info: "+str((x.shape, x.max(), x.min())))

def run_experiments(finetune, filter_lengths, nb_filter, lr, pooling, kernel_l2_regularization, other_params):
    assert (type(other_params)), type(other_params)
    global embeddings_matrix, X_train, y_train, X_test, y_test

    maxlen = X_train.shape[1]
    (vocab_size, dimensions) = embeddings_matrix.shape
    model, params = get_model(
        maxlen=maxlen, dimensions=dimensions, finetune=finetune, vocab_size=vocab_size,
        filter_lengths = filter_lengths, nb_filter = nb_filter, weights=[embeddings_matrix],
        dropout_rate=0.5, kernel_l2_regularization=kernel_l2_regularization,
        lr=lr, pooling=pooling)
    params = add_defaults(params, other_params)
    # TODO add other params here params['embeddingspath'] = 

    results_dir = '/tmp/yo/foodborne/results/test/'
    with get_archiver(datadir='/tmp/yo/foodborne/results') as temp, get_archiver() as a:
    # with get_archiver() as a:

        with open(a.getFilePath('hyperparameters.json'), 'w') as f:
            json.dump(params, f, indent=2)

        with open(a.getFilePath('model.json'), 'w') as f:
            f.write(model.to_json(indent=2))

        stdout = sys.stdout
        with open(a.getFilePath('summary.txt'), 'w') as sys.stdout:
            model.summary()
        sys.stdout = stdout

        plot_model(model, to_file=a.getFilePath('model.png'), show_shapes=True, show_layer_names=True)

        modelpath = temp.getFilePath('weights.hdf5')
        earlystopping = EarlyStopping(monitor=c.monitor, patience=c.patience, verbose=0)
        modelcheckpoint = ModelCheckpoint(modelpath, monitor=c.monitor, save_best_only=True, verbose=0)
        logger.info('starting training')
        h = model.fit(X_train, y_train, batch_size=c.batch_size, epochs=c.epochs, verbose=0,
            validation_split=0.2, callbacks=[earlystopping, modelcheckpoint])
        logger.info('ending training')

        save_history(h, a.getDirPath())

    return

