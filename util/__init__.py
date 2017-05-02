
from .experiment_utils import *
from .code_utils import *
from .string_utils import *

def fill_dict(d1, d2):
     d22 = dict(d2)
     d11 = dict(d1)
     d22.update(d11)
     d11.update(d22)
     return d11

