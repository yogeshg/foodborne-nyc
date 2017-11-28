# https://github.com/yogeshg/artistic-styles/blob/master/src/archiver.py

import os
import shutil
import pytz
import datetime as dt
from collections import OrderedDict
import sh

import json
import logging
logger = logging.getLogger(__name__)

logging.getLogger("sh.command").setLevel(logging.WARNING)

from contextlib import contextmanager

def ensureDir(p):
    if(not os.path.isdir(p)):
        logger.debug('creating directory: '+p)
        os.makedirs(p)
    return

def getTs():
    return dt.datetime.now(pytz.timezone('US/Eastern')).strftime('%Y%m%d_%H%M%S')
    # return 'test'

def cleanDir(p):
    logger.debug('cleaning directory: '+p)
    ensureDir(p)
    shutil.rmtree(p)
    ensureDir(p)
    return

def archiveDir(CURRDIR, ARCHIVE, suffix=''):
    timestamp = getTs()
    archive_name = timestamp + suffix
    wd, zd = os.path.split(CURRDIR)

    moved_dir = os.path.join(wd, archive_name)
    shutil.move(CURRDIR, moved_dir)
    wd, zd = os.path.split(moved_dir)

    logger.debug('archiving directory: '+str((wd,zd)) )
    
    # create archive
    ensureDir(ARCHIVE)
    archivePath = os.path.join(ARCHIVE, archive_name)
    st = shutil.make_archive( archivePath, 'tar', wd, zd)

    logger.info('archived directory: '+str(st) )
    return

# WORKINGDIR = os.getcwd()
# DATADIR = os.path.join(WORKINGDIR, 'data')
# CURRDIR = os.path.join(DATADIR, 'current')
# ARCHIVE = os.path.join(DATADIR, 'archive')

DEFAULT_CURRDIR = 'current_'+getTs()

class Archiver(object):
    def __init__(self, datadir=None, currdir=None, archive=None, writeInfoFile=True, suffix=''):
        if(datadir is None):
            datadir = os.path.join(os.getcwd(), 'data')
        if(currdir is None):
            currdir = os.path.join(datadir, DEFAULT_CURRDIR)
        if(archive is None):
            archive = os.path.join(datadir, 'archive')
        self.DATADIR = datadir
        self.CURRDIR = currdir
        self.ARCHIVE = archive
        self.writeInfoFile = writeInfoFile
        self.suffix = suffix
        self.info = OrderedDict()
        self.info['init'] = getTs()
        hostname = 'unknown'
        try:
            hostname = str(sh.hostname()).strip()
        except Exception as e:
            pass
        self.info['hostname']=hostname

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

    def getDirPath(self):
        return self.CURRDIR

    def getFilePath(self, p):
        return os.path.join(self.CURRDIR, p)

    def close(self):
        self.info['close'] = getTs()
        if(self.writeInfoFile):
            with open(self.getFilePath('archiver.json'), 'w') as f:
                json.dump(self.info, f, indent=2)
        archiveDir(self.CURRDIR, self.ARCHIVE, suffix=self.suffix)
        return

@contextmanager
def get_archiver(datadir=None, currdir=None, archive=None, suffix=''):
    a = Archiver(datadir=datadir, currdir=currdir, archive=archive, suffix=suffix)
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
