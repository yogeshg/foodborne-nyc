import sys
import numpy as np
import foodbornenyc.util.util as u

import logging
logging.basicConfig(level = logging.INFO, format=
        '%(asctime)s:%(levelname)s:%(name)s:%(threadName)s:line %(lineno)d: %(message)s')

from keras.preprocessing import sequence
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Embedding, GlobalMaxPooling1D, Convolution1D, merge
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.visualize_util import plot
from keras.optimizers import Adam
from datasets import yelp

from archiver import get_archiver

import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = u.get_logger(__name__)

def get_conv_stack(input_layer, nb_filter, filter_lengths, activation):
    layers = [Convolution1D(nb_filter=nb_filter, filter_length=f,
            border_mode='same', activation=activation,
            subsample_length=1)(input_layer) for f in filter_lengths]
    if (len(layers) <= 0):
        return input_layer
    elif (len(layers) == 1):
        return layers[0]
    else:
        return merge(layers, mode='concat')

def get_model(maxlen=964, dimensions=200, finetune=False, vocab_size=1000,
            pooling='max', filter_lengths=(), nb_filter=0, weights=None,
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
    assert(pooling in ['max', 'average', 'logsumexp']), '{} not in {}'.format(str(pooling), str(['max', 'average', 'logsumexp']))
    assert (all([x in (1,2,3) for x in filter_lengths])), '{} not in {}'.format(str(filter_lengths), str((1,2,3)))
    assert (type(nb_filter)==int), type(nb_filter)
    params = {k:v for k,v in locals().iteritems() if k!='weights'}

    print params # TODO print into file

    doc_input = Input(shape=(maxlen,), dtype='int32')
    embedding_layer = Embedding(vocab_size, dimensions, weights=weights, input_length=maxlen, trainable=finetune)
    y = embedding_layer(doc_input)
    y = get_conv_stack(y, nb_filter, filter_lengths, 'relu')
    assert(pooling=='max'), '{} not implemented yet'.format(pooling)
    y = GlobalMaxPooling1D()(y)
    y = Dense(1, activation='sigmoid')(y)

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

def load_data(full_data=False):
    global embeddings_matrix, X_train, y_train, X_validate, y_validate, X_test, y_test

    if (full_data):
        datapath = '/tmp/yo/foodborne/yelp_labelled.csv'
        indexpath = '/tmp/yo/foodborne/vocab_yelp.txt'
        embeddingspath = '/tmp/yo/foodborne/vectors_yelp.txt'
    else:
        datapath = '/tmp/yo/foodborne/yelp_labelled_sample.csv'
        indexpath = '/tmp/yo/foodborne/vocab_yelp_sample.txt'
        embeddingspath = '/tmp/yo/foodborne/vectors_yelp_sample.txt'


    embeddings_matrix = yelp.load_embeddings_matrix(indexpath, embeddingspath)
    ((X, y), (X_test, y_test), _) = yelp.load_devset_testset_index(datapath, indexpath, ratio_dev_test=0.8)
    ratio_train_validate = 0.8
    ((X_train, y_train), (X_validate, y_validate)) = yelp.cutXY((X, y), ratio_train_validate)

    for x in (X_train, y_train, X_validate, y_validate, X_test, y_test):
        logger.debug("shape and info: "+str((x.shape, x.max(), x.min())))



def run_experiments(finetune=False, filter_lengths = None, nb_filter = None):
    global embeddings_matrix, X_train, y_train, X_validate, y_validate, X_test, y_test

    maxlen = X_train.shape[1]
    (vocab_size, dimensions) = embeddings_matrix.shape
    model, params = get_model(
        maxlen=maxlen, dimensions=dimensions, finetune=finetune, vocab_size=vocab_size,
        filter_lengths = filter_lengths, nb_filter = nb_filter, weights=[embeddings_matrix])

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

        plot(model, to_file=a.getFilePath('model.png'), show_shapes=True, show_layer_names=True)

        modelpath = temp.getFilePath('weights.hdf5')
        callbacks = [
            EarlyStopping(monitor='val_acc', patience=500, verbose=0),
            ModelCheckpoint(modelpath, monitor='val_acc',
                save_best_only=True, verbose=0),
        ]
        h = model.fit(X_train, y_train, batch_size=256, nb_epoch=3000,
            validation_data=(X_validate, y_validate), callbacks=callbacks)

        save_history(h, a.getDirPath())

    return

def main():
    experiment_id = 0
    experiments_to_run = map(int, sys.argv[1:])
    load_data(full_data=True)
    for finetune in (False,):
        for lr in (1e-3, 1e-4, 1e-5):
            for nb_filter in (5,10,25,50,75,100):
                for filter_lengths_size in range(4):
                    filter_lengths = tuple((x+1 for x in range(filter_lengths_size)))
                    if(experiment_id in experiments_to_run):
                      try:
                        logging.info('running experiment_id: {}'.format(experiment_id))
                        run_experiments(finetune=finetune, filter_lengths = filter_lengths, nb_filter = nb_filter)
                      except Exception as e:
                        logging.exception(e)
                    experiment_id += 1
    return

if __name__ == '__main__':
    main()


