import csv
import numpy as np
# import spacy
from util.preprocessing import sequence
import logging
import util
from profilehooks import profile
import json

logger = logging.getLogger(__name__)

class LazyIndexer():

    def __init__(self):
        self._index2tokens = []
        self._tokens2index = {}

    def get_index(self, token):
        if not self._tokens2index.has_key(token):
            self._tokens2index[token] = len(self._index2tokens)
            self._index2tokens.append(token)
        return self._tokens2index[token]

    def get_token(self, index):
        if index >= len(self._index2tokens):
            return 'NONE'
        else:
            return self._index2tokens[index]

 
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
        # self.cache = util.FileDict('.spacy.en.cache.json', 1000)

    def get_tokens(self, line):
        # if not self.cache.has_key(line):
        #     tokens = [x.text for x in self.sp(util.xuni(line)) if x.pos_!='PUNCT']
        #     self.cache[line] = tokens
        # else:
        #     tokens = self.cache[line]
        tokens = [x.text for x in self.sp(util.xuni(line)) if x.pos_!='PUNCT']
        return tokens

    def get_preprocessed(self, line):
        tokens = self.get_tokens(line)
        line2 = util.xuni(" ".join(tokens))
        return line2


class Embeddings():

    def __init__(self, path, indexer, head = None):
        assert isinstance(indexer, LazyIndexer)
        self.path = path
        self.indexer = indexer
        self._head = head
        self.dimensions = None
        self._sample = 100
        with open(self.path, 'rb') as f:
            for i, text in enumerate(f):
                line = text.split()
                token = line[0]
                weights = map(float, line[1:])
                if self.dimensions is None:
                    self.dimensions = len(weights)
                else:
                    message = 'expecting {} but got {}'.format(self.dimensions, len(weights))
                    assert self.dimensions == len(weights), message
                if i > self._sample:
                    break

    def get_embeddings_matrix(self):
        vocab_size = len(self.indexer._index2tokens)
        embeddings_matrix = np.random.standard_normal(size=(vocab_size, self.dimensions))

        num_embeddings_found = 0
        num_embeddings_not_found = len(self.indexer._index2tokens)
        embeddings_not_found = set(self.indexer._index2tokens)
        self.dimensions = None

        with open(self.path, 'rb') as f:
            for text in f:
                line = text.split()
                token = line[0]

                if self.indexer._tokens2index.has_key(token):
                    weights = map(float, line[1:])
                    if(num_embeddings_found == 0):
                        self.dimensions = len(weights)
                    else:
                        message = 'expecting {} but got {}'.format(self.dimensions, len(weights))
                        assert self.dimensions == len(weights), message
                    index = self.indexer._tokens2index[token]
                    embeddings_matrix[index] = weights
                    num_embeddings_found += 1
                    num_embeddings_not_found -= 1
                    embeddings_not_found.discard(token)
                    if not self._head is None:
                        if num_embeddings_found == self._head:
                            logger.info('head limit reached, not reading more embeddings')
                            break

        logger.info('embeddings found for {}/{} tokens, not found for {}'.format(
            num_embeddings_found, vocab_size, num_embeddings_not_found))
        with open('.embeddings.notfound.json', 'w') as f:
            json.dump({'embeddings_not_found': map(str, embeddings_not_found)}, f, indent=0)
        return embeddings_matrix


class LoaderUnbiased():
    '''
    reads reviews from the yelp_official module, indexes using an indexer
    indexer : to convert lines (list of tokens) into indexes
    load_data : reads the csv and converts words to list of indexes into X, y
                makes into a numpy array if dtype is specified
                returns training and testing data
    '''

    SILVER_SIZE = 1000

    def __init__(self, dataset, datapath, indexer, preprocessor ):
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
                    index_vectors = map(self.indexer.get_index, tokens)
                    X.append( index_vectors )
                    maxlen_data = max(maxlen_data, len(index_vectors))
                except Exception as e:
                    logger.info(str(tokens))
                    logger.exception(e)
            return (X, maxlen_data)

        # create all data vectors, {X, y, w} for {train, test}
        X_train, maxlen_train = apply_preprocess(data_dict['train_data']['text'])
        y_train = data_dict['train_data']['is_foodborne']
        z_train = data_dict['train_data']['is_biased']
        w_train = calc_train_importance_weights(z_train, data_dict['U'])
        X_test, maxlen_test = apply_preprocess(data_dict['test_data']['text'])
        y_test = data_dict['test_data']['is_foodborne']
        z_test = data_dict['test_data']['is_biased']
        w_test = calc_train_importance_weights(z_test, data_dict['U'])

        # self.pp.cache.dump()

        # log shapes
        logging.debug('length of X_train: {}, y_train: {}, w_train: {}, z_train: {}'.format(
            len(X_train), len(y_train), len(w_train), len(z_train)))
        logging.debug('length of X_test: {}, y_test: {}, w_test: {}, z_test: {}'.format(
            len(X_test), len(y_test), len(w_test), len(z_test)))

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
            z_train = np.array(z_train, dtype=dtype)

            X_test = np.array(X_test, dtype=dtype)
            y_test = np.array(y_test, dtype=dtype)
            w_test = np.array(w_test, dtype=dtype)
            z_test = np.array(z_test, dtype=dtype)

        return ((X_train, y_train, w_train, z_train), (X_test, y_test, w_test, z_test))


def get_data(dataset, data_path, embeddings_path):
    preprocessor = Preprocessor()
    indexer = LazyIndexer()
    loader = LoaderUnbiased(dataset, data_path, indexer, preprocessor)

    assert len(indexer._index2tokens) == 0
    (devset, testset) = loader.load_data(dtype=np.float32)
    assert len(indexer._index2tokens) > 0

    ((X, y, w, z), (X_test, y_test, w_test, z_test)) = (devset, testset)
    logging.info("shape of training data (X, y, w, z): ({}, {}, {}, {})".format(X.shape, y.shape, w.shape, z.shape))
    logging.info("shape of test data (X, y, w, z): ({}, {}, {}, {})".format(X_test.shape, y_test.shape, w_test.shape, z_test.shape))
    logging.info("length of index2tokens: {}".format(len(indexer._index2tokens)))

    embeddings = Embeddings(embeddings_path, indexer=indexer)
    embeddings_matrix = embeddings.get_embeddings_matrix()
    logging.info("shape of embeddings_matrix: {}".format(embeddings_matrix.shape))

    return devset, testset, embeddings_matrix


@profile(immediate=True)
def test2():
    logging.basicConfig(level=logging.DEBUG, format=
    '%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:line %(lineno)d: %(message)s')

    dataset = 'yelp.silver'
    data_path = '/home/yogi/data/hdd550-data/fbnyc/yelp_data'
    embeddings_path = '/home/yogi/data/hdd550-data/fbnyc-conv/glove.840B.300d.txt'

    get_data(dataset, data_path, embeddings_path)


@profile(immediate=True)
def test1():
    logging.basicConfig(level=logging.DEBUG, format=
    '%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:line %(lineno)d: %(message)s')

    dataset = 'twitter.silver'
    data_path = '/home/yogi/data/hdd550-data/fbnyc/twitter_data'
    embeddings_path = '/home/yogi/data/hdd550-data/fbnyc-conv/glove.twitter.27B.200d.txt'

    get_data(dataset, data_path, embeddings_path)


if __name__ == '__main__':
    test2()
