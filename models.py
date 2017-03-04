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

    datapath = '/tmp/yo/foodborne/yelp_labelled_sample.txt'
    indexpath = '/tmp/yo/foodborne/vocab_yelp_sample.txt'
    embeddingspath = '/tmp/yo/foodborne/vectors_yelp_sample.txt'
    (X, y) = yelp.load_data(datapath, indexpath, embeddingspath)
    (samples, maxlen, dimensions) = X.shape
    model = get_model(maxlen=maxlen, dimensions=dimensions)
    test_samples = samples // 10
    h = model.fit(X[:-test_samples], y[:-test_samples], batch_size=10, nb_epoch=10, validation_data=(X[-test_samples:], y[-test_samples:]))
    return h

def test2():

    datapath = '/tmp/yo/foodborne/yelp_labelled_sample.txt'
    indexpath = '/tmp/yo/foodborne/vocab_yelp.txt'
    embeddingspath = '/tmp/yo/foodborne/vectors_yelp.txt'
    (X, y) = yelp.load_data(datapath, indexpath, embeddingspath)
    (samples, maxlen, dimensions) = X.shape
    model = get_model(maxlen=maxlen, dimensions=dimensions)
    test_samples = samples // 10
    h = model.fit(X[:-test_samples], y[:-test_samples], batch_size=10, nb_epoch=10, validation_data=(X[-test_samples:], y[-test_samples:]))
    return h








