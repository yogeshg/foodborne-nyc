import numpy as np
# import spacy
from util.preprocessing import sequence
import logging
import util
from profilehooks import profile
import json
from sklearn.model_selection import StratifiedShuffleSplit

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
        tokens = [x.text.lower() for x in self.sp(util.xuni(line)) if x.pos_!='PUNCT']
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
    TEST_SPLIT_DATE_STR = None
    VALIDATION_SPLIT = 0.2

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

    def load_data(self, dtype=np.float32, maxlen=None):
        logger.info('loading data for {} from {}'.format('.'.join([self.dataset_media, self.dataset_regime]), self.datapath))
        from datasets.experiments.baseline_experiment_util import setup_baseline_data, calc_importance_weights
        data_dict = setup_baseline_data(
            dataset=self.dataset_media, data_path=self.datapath,
            test_regime=self.dataset_regime, train_regime=self.dataset_regime,
            silver_size=self.SILVER_SIZE, test_split_date_str=self.TEST_SPLIT_DATE_STR)

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

        # create all data vectors, {X, y, w} for {training, validation, testing}
        # dev = train + valid
        X_dev, maxlen_dev = apply_preprocess(data_dict['train_data']['text'])
        y_dev = data_dict['train_data']['is_foodborne']
        z_dev = data_dict['train_data']['is_biased']

        X_testing, maxlen_testing = apply_preprocess(data_dict['test_data']['text'])
        y_testing = data_dict['test_data']['is_foodborne']
        z_testing = data_dict['test_data']['is_biased']
        w_testing = calc_importance_weights(z_testing, data_dict['all_B_over_U'])

        # self.pp.cache.dump()

        # apply transformations
        if(maxlen is None):
            maxlen = max(maxlen_dev, maxlen_testing)

        X_dev = sequence.pad_sequences(X_dev, maxlen=maxlen)
        X_testing = sequence.pad_sequences(X_testing, maxlen=maxlen)

        # make them numpy arrays of a certain dtype
        X_dev = np.array(X_dev, dtype=dtype)
        y_dev = np.array(y_dev, dtype=dtype)
        z_dev = np.array(z_dev, dtype=dtype)

        X_testing = np.array(X_testing, dtype=dtype)
        y_testing = np.array(y_testing, dtype=dtype)
        z_testing = np.array(z_testing, dtype=dtype)
        w_testing = np.array(w_testing, dtype=dtype)

        folds = StratifiedShuffleSplit(n_splits=1, test_size=self.VALIDATION_SPLIT, random_state=1991)
        label_bias_tuples = ['{},{}'.format(y, b) for y, b in zip(y_dev, z_dev)]
        training_idx, validation_idx = list(folds.split(np.zeros(len(z_dev)), label_bias_tuples))[0]

        X_training = X_dev[training_idx]
        y_training = y_dev[training_idx]
        z_training = z_dev[training_idx]
        w_training = calc_importance_weights(z_training, data_dict['all_B_over_U'])
        w_training = np.array(w_training, dtype=dtype)

        X_validation = X_dev[validation_idx]
        y_validation = y_dev[validation_idx]
        z_validation = z_dev[validation_idx]
        w_validation = calc_importance_weights(z_validation, data_dict['all_B_over_U'])
        w_validation = np.array(w_validation, dtype=dtype)

        return (
            (X_training, y_training, w_training, z_training),
            (X_validation, y_validation, w_validation, z_validation),
            (X_testing, y_testing, w_testing, z_testing)
        )

def get_data(dataset, data_path, embeddings_path):
    preprocessor = Preprocessor()
    indexer = LazyIndexer()
    loader = LoaderUnbiased(dataset, data_path, indexer, preprocessor)

    assert len(indexer._index2tokens) == 0
    (training_set, validation_set, testing_set) = loader.load_data(dtype=np.float32)
    assert len(indexer._index2tokens) > 0

    shapes_of_dataset = lambda (X1, y1, w1, z1): (X1.shape, y1.shape, w1.shape, z1.shape)
    message  = "shape of {} data (X, y, w, z): {}"
    logging.info(message.format('training', shapes_of_dataset(training_set)))
    logging.info(message.format('validation', shapes_of_dataset(validation_set)))
    logging.info(message.format('testing', shapes_of_dataset(testing_set)))

    logging.info("length of index2tokens: {}".format(len(indexer._index2tokens)))

    embeddings = Embeddings(embeddings_path, indexer=indexer)
    embeddings_matrix = embeddings.get_embeddings_matrix()
    logging.info("shape of embeddings_matrix: {}".format(embeddings_matrix.shape))

    return training_set, validation_set, testing_set, embeddings_matrix


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
    test1()
    test2()
