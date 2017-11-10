import sh

import logging
logging.getLogger('sh.command').setLevel(logging.WARNING)
logging.basicConfig(level = logging.DEBUG, format=
        '%(asctime)s:%(levelname)s:%(name)s:%(threadName)s:line %(lineno)d: %(message)s')

def save_code():
    try:
        o = sh.git.diff('--cached', '--name-only', '--exit-code')
    except Exception as e:
        sh.git.commit('-m', 'auto commit staged files')
    try:
        o = sh.git.diff('--name-only', '--exit-code')
    except Exception as e:
        sh.git.commit('-am', 'auto commit tracked files')
    o = sh.git('rev-parse', 'HEAD').strip()
    return o

def assert_type(argument, typename):
    message = 'got type:{} but expected type:{}'
    t1 = type(argument)
    assert t1==typename, message.format(str(t1), str(typename))

def assert_in(argument, possibilities):
    message = 'got: {} but possibilities: {}'
    assert argument in possibilities, message.format(str(argument), str(possibilities))

