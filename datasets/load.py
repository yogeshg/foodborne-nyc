import csv
import numpy as np
# import spacy
from keras.preprocessing import sequence
import logging
import util

logger = logging.getLogger(__name__)

class Index():
    '''
    This class stores the mappings between tokens to index
    indexpath : where the index file is stored, each line has a word and index
                line number serves as the index of the word
    unknown_index : this what we would represent an unknown word as
    get_index : token (integer) -> index (integer)
    get_token : index (string) -> token (string)
    '''
    def __init__(self, indexpath, unknown_index=-1):
        self.index2tokens = []
        self.tokens2index = {}
        self.indexpath = indexpath
        self.unknown_index = unknown_index
        logger.info('reading index from file, '+indexpath)
        with open(indexpath, 'rb') as f:
            for text in f:
                line = text.split()
                token = line[0]
                frequency = int(line[1])
                self.index2tokens.append(token)
                self.tokens2index[token] = len(self.index2tokens)-1

    def get_index(self, token):
        return self.tokens2index.get(token, self.unknown_index)

    def get_token(self, index):
        if index not in range(len(self.index2tokens)):
            return 'NONE'
        else:
            return self.index2tokens[index]

class Preprocessor():
    '''
    We use spacy to convert words to tokens
    install spacy and download the data file to ensure this works
    spacy automatically breaks the words based on punctuations and white spaces
    right now, we exclude all the punctuations and consider only words
    get_tokens : 'a string like this' -> ['a', 'string', 'like', 'this']
    get_preprocessed : "a string that's like this. or this!" -> "a string that 's like this or this"
    '''
    def __init__(self):
        import spacy
        self.sp = spacy.load('en')

    def get_tokens(self, line):
        tokens = [x.text for x in self.sp(util.xuni(line)) if x.pos_!='PUNCT']
        return tokens

    def get_preprocessed(self, line):
        tokens = self.get_tokens(line)
        line2 = util.xuni(" ".join(tokens))
        return line2

class Embeddings():
    """
    Deals with the word embeddings
    embeddingspath : path of the file that stores embeddings
                     first line contains m  and n separted by space
                     following m lines each contain token followed by n floats
    indexer : this is required to map each token to an index
    embeddings : token -> list of floats, reload() refreshes this
    get_embeddings : document -> list of embeddings (list of floats)
    get_embeddings_matrix : list of list of embeddings (list of floats)
    """
    def __init__(self, embeddingspath, indexer, size=None, vocab_size=None):
        self.embeddingspath = embeddingspath
        self.embeddings = {}
        self.indexer = indexer
        self.size = size
        self.vocab_size = vocab_size
        self.reload()

    def reload(self):
        logger.info('loading embeddings from file, '+self.embeddingspath)
        with open(self.embeddingspath, 'rb') as f:
            (vocab_size, size) = map(int, f.readline().split())

            if self.size is None:
                self.size = size
            assert self.size == size
            if self.vocab_size is None:
                self.vocab_size = vocab_size
            assert self.vocab_size == vocab_size

            for text in f:
                line = text.split()
                token = line[0]
                weights = [float(x) for x in line[1:]]
                if(self.size is None):
                    self.size = len(weights)
                else:
                    assert (self.size==len(weights)), 'expecting {} but got {}'.format(self.size, len(weights))
                self.embeddings[token] = weights
            assert self.vocab_size == len(self.embeddings)
        return

    def get_embeddings(self, document):
        # document is a list of indices from indexer
        return [self.embeddings[self.indexer.index2tokens[w]] for w in document]
    
    def get_embeddings_matrix(self, corpus):
        # corpus is a list of documents
        return [self.get_embeddings(d) for d in corpus]
    
    def set_embeddings_matrix(self, embedding_matrix, fname):
        # need to create token -> vector
        with open(fname, 'w') as f:
            f.write(' '.join(map(str,embedding_matrix.shape)))
            f.write('\n')
            for i,v in enumerate(embedding_matrix):
                k = self.indexer.get_token(i)
                f.write(' '.join([k]+[str(x) for x in v]))
                f.write('\n')
        

def cutXY(xy, ratio):
    (x,y) = xy
    assert len(x) == len(y), 'lengths of x ({}) and y ({}) should be the same'.format(len(x), len(y))
    cut = int(ratio * len(x))
    return ((x[:cut], y[:cut]),(x[cut:], y[cut:]))

class LoaderOld():
    '''
    reads reviews from a filepath, indexes using an indexer
    filepath : file where reviews are stored, csv file
                columns are 'data', 'label'
                each line contains quoted string as data and an integer label
    indexer : to convert lines (list of tokens) into indexes
    load_data : reads the csv and converts words to list of indexes into X, y
                makes into a numpy array if dtype is specified
                returns training and testing data
    '''
    def __init__(self, filepath, indexer, preprocessor=None ):
        self.filepath = filepath
        self.pp = preprocessor
        if( self.pp is None):
            self.pp = Preprocessor()
        self.indexer = indexer

    def load_data(self, dtype=None, maxlen=None, ratio_dev_test=0.8):
        X = []
        y = []
        logger.info('loading data from file, '+self.filepath)
        with open(self.filepath, 'rb') as f:
            reader = csv.reader(f)
            ('data', 'label')==reader.next()
            for line in reader:
                (data, label) = line
                tokens = self.pp.get_tokens(data)
                try:
                    index_vectors = [self.indexer.get_index(x) for x in tokens]
                    X.append( index_vectors )
                    y.append( int(label) )
                except Exception as e:
                    logger.info(str(tokens))
                    logger.exception(e)
        X = sequence.pad_sequences(X, maxlen=maxlen)

        if(not dtype is None):
            X = np.array(X, dtype=dtype)
            y = np.array(y, dtype=dtype)

        return cutXY((X, y), ratio_dev_test)

class LoaderUnbiased():
    '''
    reads reviews from the yelp_official module, indexes using an indexer
    indexer : to convert lines (list of tokens) into indexes
    load_data : reads the csv and converts words to list of indexes into X, y
                makes into a numpy array if dtype is specified
                returns training and testing data
    '''

    SILVER_SIZE = 1000

    def __init__(self, dataset, datapath, indexer, preprocessor=None ):
        dataset_media, dataset_regime = dataset.split('.')
        util.assert_in(dataset_media, ['yelp', 'twitter'])
        self.dataset_media = dataset_media
        self.dataset_regime = dataset_regime

        self.datapath = datapath
        self.pp = preprocessor
        if( self.pp is None):
            self.pp = Preprocessor()
        self.indexer = indexer

    def load_data(self, dtype=None, maxlen=None):
        logger.info('loading data for {} from {}'.format('.'.join([self.dataset_media, self.dataset_regime]), self.datapath))
        from datasets.experiments.baseline_experiment_util import setup_baseline_data, calc_train_importance_weights
        data_dict = setup_baseline_data(dataset=self.dataset_media, data_path=self.datapath,
                        test_regime=self.dataset_regime, train_regime=self.dataset_regime, silver_size=self.SILVER_SIZE)

        def apply_preprocess(data_x):
            X = []
            maxlen_data = 0
            for data in data_x: 
                data_str = util.xstr(data)
                tokens = self.pp.get_tokens(data_str)
                try:
                    index_vectors = [self.indexer.get_index(x) for x in tokens]
                    X.append( index_vectors )
                    maxlen_data = max(maxlen_data, len(index_vectors))
                except Exception as e:
                    logger.info(str(tokens))
                    logger.exception(e)
            return (X, maxlen_data)

        # create all data vectors, {X, y, w} for {train, test}
        X_train, maxlen_train = apply_preprocess(data_dict['train_data']['text'])
        y_train = data_dict['train_data']['is_foodborne']
        w_train = calc_train_importance_weights(data_dict['train_data']['is_biased'] , data_dict['U'])
        X_test, maxlen_test = apply_preprocess(data_dict['test_data']['text'])
        y_test = data_dict['test_data']['is_foodborne']
        w_test = calc_train_importance_weights(data_dict['test_data']['is_biased'] , data_dict['U'])

        # log shapes
        logging.debug('length of X_train: {}, y_train: {}, w_train: {}'.format(len(X_train), len(y_train), len(w_train)))
        logging.debug('length of X_test: {}, y_test: {}, w_test: {}'.format(len(X_test), len(y_test), len(w_test)))

        # apply transformations
        if(maxlen is None):
            maxlen = max(maxlen_train, maxlen_test)
        X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
        X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

        # make them numpy arrays of a certains dtype iff specified
        if(not dtype is None):
            X_train = np.array(X_train, dtype=dtype)
            y_train = np.array(y_train, dtype=dtype)
            w_train = np.array(w_train, dtype=dtype)
            X_test = np.array(X_test, dtype=dtype)
            y_test = np.array(y_test, dtype=dtype)
            w_test = np.array(w_test, dtype=dtype)

        return ((X_train, y_train, w_train), (X_test, y_test, w_test))

def load_devset_testset_index(dataset, indexpath, maxlen=None, dtype=np.float32, datapath=None):
    util.assert_type(dataset, str)

    if datapath is None:
        if 'yelp' == dataset.split('.')[0]:
            datapath = '~/data/hdd550-data/fbnyc/yelp_data/'
        elif 'twitter' == dataset.split('.')[0]:
            datapath = '~/data/hdd550-data/fbnyc/twitter_data/'
        else:
            logging.info('Cannot infer datapath from dataset '+dataset)
    
    logger.debug('loading {} data from path: {} and index: {}'.format(dataset, datapath, str(indexpath)))
    global p, i, l
    p = Preprocessor()
    i = Index(indexpath, unknown_index=0)
    l = LoaderUnbiased(dataset, datapath, i, p)

    (devset, testset) = l.load_data(dtype=dtype)

    return (devset, testset, i.index2tokens)

def load_embeddings_matrix(indexpath, embeddingspath):
    logger.debug('loading yelp embeddings: '+str(locals()))
    indexer = Index(indexpath, unknown_index=0)
    embedder = Embeddings(embeddingspath, indexer)
    message = 'embeddings ({}) and index ({}) size should match'.format(embedder.vocab_size, len(indexer.index2tokens))
    assert(embedder.vocab_size == len(indexer.index2tokens)), message

    m = np.zeros((embedder.vocab_size, embedder.size))
    for i,w in enumerate(indexer.index2tokens):
        m[i] = embedder.embeddings[w]
    return m

def test():

    dataset = 'yelp.silver'
    indexpath = 'data/vocab_yelp_sample.txt'
    embeddingspath = 'data/vectors_yelp_sample.txt'
    ((X, y, w), (X_test, y_test, w_test), index2tokens) = load_devset_testset_index(dataset, indexpath)
    logging.info("shape of training data (X, y, w): ({}, {}, {})".format(X.shape, y.shape, w.shape))
    logging.info("shape of test data (X, y, w): ({}, {}, {})".format(X_test.shape, y_test.shape, w_test.shape))
    logging.info("length of index2tokens: {}".format(len(index2tokens)))
    m = load_embeddings_matrix(indexpath, embeddingspath)
    logging.info("shape of embeddings_matrix: {}".format(m.shape))

if __name__ == '__main__':
    test()
