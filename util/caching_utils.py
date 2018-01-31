import json
import os
import errno
import re

import collections
import functools

import cPickle
import logging
logger = logging.getLogger(__name__)

WORD_REGEX = re.compile(r'[\w]+')
def fsSafeString(text):
    logger.debug('making filesystem safe string: {}'.format(text))
    return "_".join(WORD_REGEX.findall(json.dumps(text)))

def ensureDir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


__idempotent_last_args = None
__idempotent_last_result = None

def idempotent(some_function):


    def wrapper(*args):
        global __idempotent_last_args, __idempotent_last_result
        if __idempotent_last_args != args:
            __idempotent_last_args = args
            result = some_function(*args)
            __idempotent_last_result = result
        return __idempotent_last_result

    return wrapper


class FileDict():
    def __init__(self, filename, dump_frequency):
        self.filename = filename
        self.num_sets = 0
        self.dump_frequency = dump_frequency
        self.dict = {}
        self.reload()

    def __del__(self):
        self.dump()

    def reload(self):
        logger.debug('reloading FileDict from file: {}'.format(self.filename))
        try:
            with open(self.filename, 'r') as f:
                dict = cPickle.load(f)
                self.dict = dict
            logger.debug('keys loaded: {}'.format(self.dict.keys()))
        except Exception as e:
            pass
            # logging.exception(e)
            # logging.warning('reload failed from file: {}'.format(self.filename))
        return

    def dump(self):
        logger.debug('dumping in {}'.format(self.filename))
        with open(self.filename, 'w') as f:
            # cPickle.dump(self, f, sort_keys=True, indent=None)
            cPickle.dump(self.dict, f)

    def __setitem__(self, i, y):
        r = self.dict[i] = y
        self.num_sets += 1
        if self.num_sets % self.dump_frequency == self.dump_frequency-1:
            self.dump()
        return r

    def __getitem__(self, i):
        return self.dict[i]



class DirDict():
    def __init__(self, dirname, dump_frequency, key_hasher=fsSafeString):
        ensureDir(dirname)
        self.dirname = dirname
        self.num_sets = 0
        self.dump_frequency = dump_frequency
        self.keyHasher = key_hasher
        self.index = FileDict(os.path.join(dirname, '.keys2file'), dump_frequency)
        self.dict = {}

    def __del__(self):
        self.dump()

    def filePath(self, filename):
        return os.path.join(self.dirname, filename)

    def __setitem__(self, key, content):
        filename = self.keyHasher(key)
        self.index[key] = filename
        self.dict[key] = content
        self.num_sets += 1
        if self.num_sets % self.dump_frequency == self.dump_frequency-1:
            self.dump()

    def dump(self):
        for key in self.dict.keys():
            filename = self.index[key]
            content = self.dict[key]
            with open(self.filePath(filename), 'w') as f:
                cPickle.dump(content, f)

    def __getitem__(self, key):
        if key in self.dict.keys():
            return self.dict[key]
        elif key in self.index.dict.keys():
            filename = self.index[key]
            with open( self.filePath(filename), 'r' ) as f:
                r = cPickle.load(f)
            self.dict[key] = r
            return r
        else:
            raise KeyError, key

class memoized(object):
   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   '''
   def __init__(self, func, cache_type=dict, cache_type_args=[], cache_type_kwargs={}):
      self.func = func
      self.cache = cache_type(*cache_type_args, **cache_type_kwargs)
      logger.info('created memoized function')

   def __call__(self, *args):
      if not isinstance(args, collections.Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         logger.warn('cannot hash args {}, thus not caching results'.format(str(args)))
         return self.func(*args)
      try:
         return self.cache[args]
      except:
         value = self.func(*args)
         self.cache[args] = value
         return value

   def __repr__(self):
      '''Return the function's docstring.'''
      return self.func.__doc__

   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)



def test_filedict():
    import uuid
    fname = '/tmp/test_file_dict_' + uuid.uuid4().get_hex()[:6]

    def set_values(fname):
        fd = FileDict(fname, 10)
        fd['a'] = 'a'
        fd['b'] = 'b'

    def assert_values(fname):
        fd = FileDict(fname, 1)
        assert fd['a'] == 'a'
        assert fd['b'] == 'b'

    set_values(fname)
    assert_values(fname)

    def set_values(fname):
        fd = FileDict(fname, 10)
        fd[(1, 2)] = 'a'
        fd[(3, 4)] = 'b'

    def assert_values(fname):
        fd = FileDict(fname, 1)
        assert fd[(1, 2)] == 'a'
        assert fd[(3, 4)] == 'b'

    set_values(fname)
    assert_values(fname)


def test_dirdict():
    import uuid
    dname = '/tmp/test_dir_dict_' + uuid.uuid4().get_hex()[:6]

    def set_values(dname):
        fd = DirDict(dname, 10)
        fd['a'] = 'a'
        fd['b'] = 'b'

    def assert_values(fname):
        fd = DirDict(fname, 1)
        assert fd['a'] == 'a'
        assert fd['b'] == 'b'

    set_values(dname)
    assert_values(dname)

def test_memoized():
    import uuid
    dname = '/tmp/test_memoized_' + uuid.uuid4().get_hex()[:6]
    mymemoized = functools.partial(memoized, cache_type=DirDict,
                                   cache_type_args=[dname, 1])
    @mymemoized
    def fibonacci(n):
        n = int(n)
        if n > 1:
            return fibonacci(str(n-1)) + fibonacci(str(n-2))
        else:
            return n

    n = fibonacci('1')
    n = fibonacci('5')

    print sorted(fibonacci.cache.dict.items())

if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.DEBUG)
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'test':
        for n, f in filter(lambda (n,f): 'test_' in n and callable(f), locals().items()):
            print(n)
            f()
