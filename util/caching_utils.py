import json


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


class FileDict(dict):
    def __init__(self, filename, dump_frequency):
        self.filename = filename
        self.num_sets = 0
        self.dump_frequency = dump_frequency
        self.reload()

    def reload(self):
        try:
            with open(self.filename, 'r') as f:
                d = json.load(f)
                dict.__init__(self, d)
        except:
            pass
        return

    def dump(self):
        with open(self.filename, 'w') as f:
            json.dump(self, f, sort_keys=True, indent=None)

    def __setitem__(self, i, y):
        r = dict.__setitem__(self, i, y)
        self.num_sets += 1
        if self.num_sets % self.dump_frequency == self.dump_frequency-1:
            self.dump()
        return r

    def __getitem__(self, i):
        return dict.__getitem__(self, i)
