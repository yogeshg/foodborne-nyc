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
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.engine.topology import Layer
from keras import backend as K
from datasets import yelp

from archiver import get_archiver

import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = u.get_logger(__name__)

def add_defaults(d1, d2):
     d22 = dict(d2)
     d11 = dict(d1)
     d22.update(d11)
     d11.update(d22)
     return d11

class LogSumExpPooling(Layer):

    def call(self, x):
        # could be axis 0 or 1
        import numpy as np
        return K.log(K.sum(K.exp(x), axis=1))

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
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    return (model, params)

def save_history(history, dirpath):
    with open(dirpath+'/training.json', 'w') as f:
        json.dump(history.params, f, indent=2)

    df = pd.DataFrame.from_dict(history.history)
    df.to_csv(dirpath+'/history.csv')
    i = df.val_acc.argmax()

    cname = 'val_acc_{:04d}'.format(i)
    df.loc[:, cname] = df.loc[i,'val_acc']
    df.loc[:, ['acc', 'val_acc', cname]].plot()
    plt.savefig(dirpath+'/acc.png')

    cname = 'val_loss_{:04d}'.format(i)
    df.loc[:, cname] = df.loc[i,'val_loss']
    df.loc[:, ['loss', 'val_loss', cname]].plot()
    plt.savefig(dirpath+'/loss.png')

    return

def load_data(datapath, indexpath, embeddingspath, testdata=False):
    global embeddings_matrix, X_train, y_train, X_validate, y_validate, X_test, y_test
    if( testdata ):
        datapath = '/tmp/yo/foodborne/yelp_labelled_sample.csv'
        indexpath = '/tmp/yo/foodborne/vocab_yelp_sample.txt'
        embeddingspath = '/tmp/yo/foodborne/vectors_yelp_sample.txt'
    
    embeddings_matrix = yelp.load_embeddings_matrix(indexpath, embeddingspath)
    ((X, y), (X_test, y_test), _) = yelp.load_devset_testset_index(datapath, indexpath)
    ratio_train_validate = 0.8
    ((X_train, y_train), (X_validate, y_validate)) = yelp.cutXY((X, y), ratio_train_validate)

    for x in (X_train, y_train, X_validate, y_validate, X_test, y_test):
        logger.debug("shape and info: "+str((x.shape, x.max(), x.min())))



def run_experiments(finetune, filter_lengths, nb_filter, lr, pooling, kernel_l2_regularization, other_params):
    assert (type(other_params)), type(other_params)
    global embeddings_matrix, X_train, y_train, X_validate, y_validate, X_test, y_test

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

        # plot_model(model, to_file=a.getFilePath('model.png'), show_shapes=True, show_layer_names=True)

        modelpath = temp.getFilePath('weights.hdf5')
        patience = 500
        if(pooling=='logsumexp'):
            patience = 1000
        callbacks = [
            EarlyStopping(monitor='val_acc', patience=patience, verbose=0),
            ModelCheckpoint(modelpath, monitor='val_acc',
                save_best_only=True, verbose=0),
        ]
        logger.info('starting training')
        h = model.fit(X_train, y_train, batch_size=256, nb_epoch=300, verbose=0,
            validation_data=(X_validate, y_validate), callbacks=callbacks)
        logger.info('ending training')

        save_history(h, a.getDirPath())

    return

def main():
    return

if __name__ == '__main__':
    main()

