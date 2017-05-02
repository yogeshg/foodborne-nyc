import logging
logger = logging.getLogger(__name__)

from keras.models import Model
from keras.layers import Input, Dense, Embedding, Dropout, GlobalMaxPooling1D, GlobalAveragePooling1D
from layers import LogSumExpPooling, get_conv_stack
from keras import regularizers
from keras.optimizers import Adam

from metrics import auc


def get_model(maxlen=964, dimensions=200, finetune=False, vocab_size=1000,
            pooling='max', kernel_sizes=(), filters=0, weights=None,
            dropout_rate=0, kernel_l2_regularization=0,
            lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0 ):
    '''
    maxlen : maximum size of each document
    dimensions : dimension of each vector
    finetune : [True, False] : weather or not to finetune word emdeddings
    vocab_size : size of the vocabulary, emdeddings layer will be this big
    pooling : ['average', 'logsumexp'] : pooling operation for word vectors in a document
    kernel_sizes : tuple : convolve using unigrams / bigrams / trigrams
    filters : None or int : number of filters for convolutional layer
    '''
    assert(type(dimensions)==int), type(dimensions)
    assert(type(maxlen)==int), type(maxlen)
    assert(type(finetune)==bool), type(finetune)
    assert(type(vocab_size)==int), type(vocab_size)
    assert(pooling in ['max', 'avg', 'logsumexp']), '{} not in {}'.format(str(pooling), str(['max', 'average', 'logsumexp']))
    assert (all([x in range(10) for x in kernel_sizes])), '{} not in {}'.format(str(kernel_sizes), str((1,2,3)))
    assert (type(filters)==int), type(filters)
    params = {k:v for k,v in locals().iteritems() if k!='weights'}

    logger.info( str(params) )

    doc_input = Input(shape=(maxlen,), dtype='int32')
    embedding_layer = Embedding(vocab_size, dimensions, weights=weights, input_length=maxlen, trainable=finetune)
    y = embedding_layer(doc_input)
    y = Dropout(dropout_rate, noise_shape=None, seed=None)(y)
    y = get_conv_stack(y, filters, kernel_sizes, 'relu', kernel_l2_regularization, dropout_rate)
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
    model.summary()
    adam = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy', auc])

    return (model, params)
