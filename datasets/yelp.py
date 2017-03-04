import numpy as np

import spacy
from keras.preprocessing import sequence

import foodbornenyc.util.util as u

import logging

logging.basicConfig(level = logging.DEBUG, format=
        '%(asctime)s:%(levelname)s:%(name)s:%(threadName)s:line %(lineno)d: %(message)s')

logger = u.get_logger(__name__)

class Index():
    def __init__(self, indexpath):
        self.index2tokens = []
        self.tokens2index = {}
        self.indexpath = indexpath
        logger.info('reading index from file, '+indexpath)
        with open(indexpath, 'rb') as f:
            for text in f:
                line = text.split()
                token = line[0]
                frequency = int(line[1])
                self.index2tokens.append(token)
                self.tokens2index[token] = len(self.index2tokens)-1

    def get_index(self, token):
        return self.tokens2index.get(token, -1)

    def get_token(self, index):
        if index not in range(len(index2tokens)):
            return 'NONE'
        else:
            return self.index2tokens[index]


class Preprocessor():
    def __init__(self, datapath='/tmp/yo/.python/spacy/data/'):
        logger.info('loading spacy from file, '+datapath)
        spacy.util.set_data_path(datapath)
        self.sp = spacy.load('en')

    def get_tokens(self, line):
        tokens = [x.text for x in self.sp(u.xuni(line)) if x.pos_!='PUNCT']
        return tokens

    def get_preprocessed(self, line):
        tokens = self.get_tokens(line)
        line2 = u.xuni(" ".join(tokens))
        return line2

class Loader():
    def __init__(self, filepath, indexer, preprocessor=None ):
        self.filepath = filepath
        self.pp = preprocessor
        if( self.pp is None):
            self.pp = Preprocessor()
        self.indexer = indexer

    def load_data(self):
        X = []
        y = []
        logger.info('loading data from file, '+self.filepath)
        with open(self.filepath, 'rb') as f:
            for text in f:
                line = text.split(",")
                label = line[0]
                text = line[1]
                tokens = self.pp.get_tokens(text)
                try:
                    index_vectors = [self.indexer.get_index(x) for x in tokens]
                    X.append( index_vectors )
                    y.append( label )
                except Exception as e:
                    logger.info(str(tokens))
                    logger.exception(e)
        return (X, y)

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

def load_data( filepath, indexpath, embeddingspath, maxlen=None, dtype=np.float32):
    logger.info('local variables: '+str(locals()))
    p = Preprocessor()
    i = Index(indexpath)
    l = Loader(filepath, i, p)
    e = Embeddings(embeddingspath, i)

    (X, y) = l.load_data()
    X = sequence.pad_sequences(X, maxlen=maxlen)
    
    V = e.get_embeddings_matrix(X)
    m = {'negative': 0, 'positive': 1}
    y = map(lambda x: m[x], y)

    y = np.array(y, dtype=dtype)
    V = np.array(V, dtype=dtype)

    return (V, y)

def test():

    datapath = '/tmp/yo/foodborne/yelp_labelled_sample.txt'
    indexpath = '/tmp/yo/foodborne/vocab_yelp_sample.txt'
    embeddingspath = '/tmp/yo/foodborne/vectors_yelp_sample.txt'
    (X, y) = load_data(datapath, indexpath, embeddingspath)
    print X.shape
    print y.shape

if __name__ == '__main__':
    test()
