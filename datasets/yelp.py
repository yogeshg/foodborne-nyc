import logging
import csv

import numpy as np

# import spacy
from keras.preprocessing import sequence

import foodbornenyc.util.util as u


logging.basicConfig(level = logging.DEBUG, format=
        '%(asctime)s:%(levelname)s:%(name)s:%(threadName)s:line %(lineno)d: %(message)s')

logger = u.get_logger(__name__)

class Index():
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
        if index not in range(len(index2tokens)):
            return 'NONE'
        else:
            return self.index2tokens[index]


class Preprocessor():
    def __init__(self, datapath='/tmp/yo/.python/spacy/data/'):
        logger.info('loading spacy from file, '+datapath)
        import spacy
        spacy.util.set_data_path(datapath)
        self.sp = spacy.load('en')

    def get_tokens(self, line):
        tokens = [x.text for x in self.sp(u.xuni(line)) if x.pos_!='PUNCT']
        return tokens

    def get_preprocessed(self, line):
        tokens = self.get_tokens(line)
        line2 = u.xuni(" ".join(tokens))
        return line2

class Embeddings():
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
        # corpus is a list of documents:
        return [self.get_embeddings(d) for d in corpus]    

def cutXY(xy, ratio):
    (x,y) = xy
    assert len(x) == len(y), 'lengths of x ({}) and y ({}) should be the same'.format(len(x), len(y))
    cut = int(ratio * len(x))
    return ((x[:cut], y[:cut]),(x[cut:], y[cut:]))

class Loader():
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

def load_devset_testset_index(datapath, indexpath, maxlen=None, dtype=np.float32, ratio_dev_test=0.8):
    logger.debug('loading yelp data and index: '+str(locals()))
    p = Preprocessor()
    i = Index(indexpath, unknown_index=0)
    l = Loader(datapath, i, p)

    (devset, testset) = l.load_data(dtype=dtype, ratio_dev_test=ratio_dev_test)

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

    datapath = '/tmp/yo/foodborne/yelp_labelled_sample.csv'
    indexpath = '/tmp/yo/foodborne/vocab_yelp_sample.txt'
    embeddingspath = '/tmp/yo/foodborne/vectors_yelp_sample.txt'
    ((X, y), (X_test, y_test), index2tokens) = load_devset_testset_index(datapath, indexpath)
    print (X.shape, y.shape)
    print (X_test.shape, y_test.shape)
    print len(index2tokens)
    m = load_embeddings_matrix(indexpath, embeddingspath)
    print m.shape

if __name__ == '__main__':
    test()
