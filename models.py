import numpy as np
import foodbornenyc.util.util as u

import logging
logging.basicConfig(level = logging.INFO, format=
        '%(asctime)s:%(levelname)s:%(name)s:%(threadName)s:line %(lineno)d: %(message)s')

from keras.preprocessing import sequence
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from datasets import yelp

from archiver import get_archiver

import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = u.get_logger(__name__)

def get_model(maxlen=964, dimensions=200, finetune = False, vocab_size=1000,
            pooling = 'average', ngrams = None, num_filters = None, embeddings_matrix=None):
    '''
    maxlen : maximum size of each document
    dimensions : dimension of each vector
    finetune : [True, False] : weather or not to finetune word emdeddings
    vocab_size : size of the vocabulary, emdeddings layer will be this big
    pooling : ['average', 'logsumexp'] : pooling operation for word vectors in a document
    ngrams : None or tuple : convolve using unigrams / bigrams / trigrams
    num_filters : None or int : number of filters for convolutional layer
    '''
    assert(type(dimensions)==int), type(dimensions)
    assert(type(maxlen)==int), type(maxlen)
    assert(type(finetune)==bool), type(finetune)
    assert(type(vocab_size)==int), type(vocab_size)
    assert(pooling in ['average', 'logsumexp']), '{} not in {}'.format(str(pooling), str(['average', 'logsumexp']))
    assert (ngrams is None) or (all([x in (1,2,3) for x in ngrams])), '{} not in {}'.format(str(ngrams), str((1,2,3)))
    assert (num_filters is None) or (type(num_filters)==int), type(num_filters)
    params = {k:v for k,v in locals().iteritems() if k!='embeddings_matrix'}

    print params # TODO print into file

    doc_input = Input(shape=(maxlen,), dtype='int32')
    embedding_layer = Embedding(vocab_size, dimensions, weights=[embeddings_matrix], input_length=maxlen, trainable=finetune)
    y = embedding_layer(doc_input)
    y = GlobalAveragePooling1D(input_shape=(maxlen, dimensions))(y)
    y = Dense(1, activation='sigmoid')(y)

    model = Model(doc_input, y)
    model.summary() # TODO print into file

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

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

def test(full_data=False):

    if(full_data):
        datapath = '/tmp/yo/foodborne/yelp_labelled.csv'
        indexpath = '/tmp/yo/foodborne/vocab_yelp.txt'
        embeddingspath = '/tmp/yo/foodborne/vectors_yelp.txt'
    else:
        datapath = '/tmp/yo/foodborne/yelp_labelled_sample.csv'
        indexpath = '/tmp/yo/foodborne/vocab_yelp_sample.txt'
        embeddingspath = '/tmp/yo/foodborne/vectors_yelp_sample.txt'

    embeddings_matrix = yelp.load_embeddings_matrix(indexpath, embeddingspath)
    ((X, y), (X_test, y_test), _) = yelp.load_devset_testset_index(datapath, indexpath)
    ratio_train_validate = 0.8
    ((X_train, y_train), (X_validate, y_validate)) = yelp.cutXY((X, y), ratio_train_validate)

    for x in (X_train, y_train, X_validate, y_validate, X_test, y_test):
        logger.debug("shape and info: "+str((x.shape, x.max(), x.min())))

    maxlen = X.shape[1]
    (vocab_size, dimensions) = embeddings_matrix.shape
    model, params = get_model(maxlen=maxlen, dimensions=dimensions,
        vocab_size=vocab_size, embeddings_matrix=embeddings_matrix)

    results_dir = '/tmp/yo/foodborne/results/test/'
    with get_archiver(datadir='/tmp/yo/foodborne/results') as temp, get_archiver() as a:

        with open(a.getFilePath('hyperparameters.json'), 'w') as f:
            json.dump(params, f, indent=2)

        with open(a.getFilePath('model.json'), 'w') as f:
            f.write(model.to_json(indent=2))

        modelpath = temp.getFilePath('weights.hdf5')
        callbacks = [
            EarlyStopping(monitor='val_acc', patience=10, verbose=0),
            ModelCheckpoint(modelpath, monitor='val_acc',
                save_best_only=True, verbose=0),
        ]
        h = model.fit(X_train, y_train, batch_size=128, nb_epoch=2000,
            validation_data=(X_validate, y_validate), callbacks=callbacks)

        save_history(h, a.getDirPath())

    return h

if __name__ == '__main__':
    test(False)
    pass
