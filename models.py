from keras.preprocessing import sequence
from keras.models import Sequential
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

def get_model(maxlen=964, dimensions=200, finetune = False,
            pooling = 'average', ngrams = None, num_filters = None):
    '''
    maxlen : maximum size of each document
    dimensions : dimension of each vector
    finetune : [True, False] : weather or not to finetune word emdeddings
    pooling : ['average', 'logsumexp'] : pooling operation for word vectors in a document
    ngrams : None or tuple : convolve using unigrams / bigrams / trigrams
    num_filters : None or int : number of filters for convolutional layer
    '''
    assert(type(dimensions)==int), type(dimensions)
    assert(type(maxlen)==int), type(maxlen)
    assert(type(finetune)==bool), type(maxlen)
    assert(pooling in ['average', 'logsumexp']), '{} not in {}'.format(str(pooling), str(['average', 'logsumexp']))
    assert (ngrams is None) or (all([x in (1,2,3) for x in ngrams])), '{} not in {}'.format(str(ngrams), str((1,2,3)))
    assert (num_filters is None) or (type(num_filters)==int), type(num_filters)
    param = locals()

    print param # TODO print into file 

    model = Sequential()
    model.add(GlobalAveragePooling1D(input_shape=(maxlen, dimensions)))
    model.add(Dense(1, activation='sigmoid'))

    model.summary() # TODO print into file 

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def save_history(history, dirpath):
    with open(dirpath+'/params.json', 'w') as f:
        json.dump(history.params, f, indent=2)
    df = pd.DataFrame.from_dict(history.history)
    df.to_csv(dirpath+'/history.csv')
    with open(dirpath+'/model.json', 'w') as f:
        f.write(history.model.to_json(indent=2))
    i = df.val_acc.argmax()
    cname = 'val_acc_{:04d}'.format(i)
    df.loc[:, cname] = df.loc[i,'val_acc']
    df.loc[:, ['acc', 'val_acc', cname]].plot()
    plt.savefig(dirpath+'/acc.png')
    cname = 'val_loss_{:04d}'.format(i)
    df.loc[:, cname] = df.loc[i,'loss_acc']
    df.loc[:, ['loss', 'val_loss', cname]].plot()
    plt.savefig(dirpath+'/loss.png')
    return

def test1():

    datapath = '/tmp/yo/foodborne/yelp_labelled_sample.csv'
    indexpath = '/tmp/yo/foodborne/vocab_yelp_sample.txt'
    embeddingspath = '/tmp/yo/foodborne/vectors_yelp_sample.txt'
    ((X, y), (X_test, y_test)) = yelp.load_data(datapath, indexpath, embeddingspath)
    (samples, maxlen, dimensions) = X.shape
    model = get_model(maxlen=maxlen, dimensions=dimensions)
    validation_samples = samples // 10

    ratio_train_validate = 0.8
    cut = int(ratio_train_validate * X.shape[0])
    ((X_train, y_train), (X_validate, y_validate)) = ((X[:cut], y[:cut]), (X[cut:], y[cut:]))
    print X_train.shape
    print y_train.shape
    print X_validate.shape
    print y_validate.shape

    h = model.fit(X_train, y_train, batch_size=10, nb_epoch=10, validation_data=(X_validate, y_validate))
    return h


def test2():

    datapath = '/tmp/yo/foodborne/yelp_labelled.csv'
    indexpath = '/tmp/yo/foodborne/vocab_yelp.txt'
    embeddingspath = '/tmp/yo/foodborne/vectors_yelp.txt'
    ((X, y), (X_test, y_test)) = yelp.load_data(datapath, indexpath, embeddingspath)
    (samples, maxlen, dimensions) = X.shape
    model = get_model(maxlen=maxlen, dimensions=dimensions)
    validation_samples = samples // 10

    ratio_train_validate = 0.8
    cut = int(ratio_train_validate * X.shape[0])
    ((X_train, y_train), (X_validate, y_validate)) = ((X[:cut], y[:cut]), (X[cut:], y[cut:]))
    print X_train.shape
    print y_train.shape
    print X_validate.shape
    print y_validate.shape
    results_dir = '/tmp/yo/foodborne/results/test/'
    with get_archiver(datadir='/tmp/yo/foodborne/results') as temp, get_archiver() as a:

        modelpath = temp.getFilePath('weights.hdf5')
        callbacks = [
            EarlyStopping(monitor='val_acc', patience=50, verbose=0),
            ModelCheckpoint(modelpath, monitor='val_acc',
                save_best_only=True, verbose=0),
        ]
        h = model.fit(X_train, y_train, batch_size=128, nb_epoch=2000,
            validation_data=(X_validate, y_validate), callbacks=callbacks)
        save_history(h, a.getDirPath())
    return h

if __name__ == '__main__':
    test2()
