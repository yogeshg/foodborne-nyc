# https://github.com/yogeshg/artistic-styles/blob/master/src/archiver.py

import os
import shutil
import pytz
import datetime as dt
from collections import OrderedDict

import json
import logging
logger = logging.getLogger(__name__)

from contextlib import contextmanager

def ensureDir(p):
    if(not os.path.isdir(p)):
        logger.info('creating directory: '+p)
        os.makedirs(p)
    return

def getTs():
    return dt.datetime.now(pytz.timezone('US/Eastern')).strftime('%Y%m%d_%H%M%S')

def cleanDir(p):
    logger.info('cleaning directory: '+p)
    ensureDir(p)
    shutil.rmtree(p)
    ensureDir(p)
    return

def archiveDir(CURRDIR, ARCHIVE):
    ensureDir(ARCHIVE)
    archivePath = os.path.join(ARCHIVE, getTs())
    wd, zd = os.path.split(CURRDIR)
    logger.info('archiving directory: '+str((wd,zd)) )
    st = shutil.make_archive( archivePath, 'tar', wd, zd)
    logger.info('archived directory: '+str(st) )
    return

# WORKINGDIR = os.getcwd()
# DATADIR = os.path.join(WORKINGDIR, 'data')
# CURRDIR = os.path.join(DATADIR, 'current')
# ARCHIVE = os.path.join(DATADIR, 'archive')

class Archiver(object):
    def __init__(self, datadir=None, currdir=None, archive=None, writeInfoFile=True):
        if(datadir is None):
            datadir = os.path.join(os.getcwd(), 'data')
        if(currdir is None):
            currdir = os.path.join(datadir, 'current')
        if(archive is None):
            archive = os.path.join(datadir, 'archive')
        self.DATADIR = datadir
        self.CURRDIR = currdir
        self.ARCHIVE = archive
        self.writeInfoFile = writeInfoFile
        self.info = OrderedDict()
        self.info['init'] = getTs()
        error_message = 'No paths should end in slash'
        assert not '/' in map(lambda x:x[-1], [self.ARCHIVE, self.DATADIR, self.CURRDIR]), error_message
        return

    def __str__(self):
        s = 'Archiver:'
        s += '\ndatadir: ' + self.DATADIR
        s += '\ncurrdir: ' + self.CURRDIR
        s += '\narchive: ' + self.ARCHIVE
        return s

    def open(self):
        cleanDir(self.CURRDIR)
        self.info['open'] = getTs()
        return

    def getFilePath(self, p):
        return os.path.join(self.CURRDIR, p)

    def close(self):
        self.info['close'] = getTs()
        if(self.writeInfoFile):
            with open(self.getFilePath('archiver.json'), 'w') as f:
                json.dump(self.info, f, indent=2)
        archiveDir(self.CURRDIR, self.ARCHIVE)
        return

@contextmanager
def get_archiver(datadir=None, currdir=None, archive=None):
    a = Archiver(datadir=datadir, currdir=currdir, archive=archive)
    a.open()
    try:
        yield a
    finally:
        a.close()

def test():
    a = Archiver()
    a.open()
    with open(a.getFilePath('a.txt'), 'w') as f:
        f.write('temp')
    a.close()
    return

if __name__ == '__main__':
    test()
