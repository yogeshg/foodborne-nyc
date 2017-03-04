from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from datasets import yelp

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

    h = model.fit(X_train, y_train, batch_size=10, nb_epoch=10, validation_data=(X_validate, y_validate))
    return h

