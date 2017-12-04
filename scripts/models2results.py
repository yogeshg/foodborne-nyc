import pandas as pd
import os
import sys
import json
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_history(d, idx):
   df = pd.read_csv(os.path.join(d, 'history.csv'), index_col='epoch')
   return df.loc[idx]

def get_json(dirname, filename):
   j = {}
   with open(os.path.join(dirname, filename+'.json')) as f:
      j = json.load(f)
   j['dirname'] = dirname
   return j

def get_all_info(d):
   record = {}
   record.update(get_json(d, 'hyperparameters'))
   record.update(get_json(d, 'training'))
   return record

def get_list_idx(l, i, default):
   if len(l) > i:
      return l[i]
   else:
      return default

assert len(sys.argv) > 2

logger.info('arguments: {}'.format(sys.argv))

DIR = sys.argv[1]
CONTAINS = sys.argv[2]
OUTPUT = get_list_idx(sys.argv, 3, 'md')
SORT_BY = get_list_idx(sys.argv, 4, 'dataset')
SELECT_COLS = get_list_idx(sys.argv, 5, None)

# dirs = [f for f in os.listdir('./view/') if 'c597c3' in f]
dirs = [os.path.join(DIR, f) for f in os.listdir(DIR) if CONTAINS in f]

logger.info('found the following folders: \n{}'.format("\n".join(dirs)))

logger.info('getting hyper parameters and training results')
df_info = pd.DataFrame([get_all_info(d) for d in dirs])

logger.info('looking up history for eah result')
df_metrics = pd.DataFrame(map(get_history, df_info.dirname, df_info.es__best_epoch.map(int))).reset_index(drop=True)

df_all = df_info.join(df_metrics)
df_all = df_all.sort_values(by=SORT_BY)
if SELECT_COLS is None:
   SELECT_COLS = ','.join(list(df_all.columns))
df_all = df_all.select(lambda c: c in SELECT_COLS.split(','), axis=1)

logger.info('sorted all information by {} and selected {} columns'.format(SORT_BY, SELECT_COLS))

if OUTPUT == 'md':
   lines = df_all.to_csv(sep='|', index=False).split("\n")
   header = "".join((map(lambda c: c if c=="|" else "-",lines[0])))
   print "\n".join([lines[0], header] + lines[1:])
elif OUTPUT == 'csv':
   print df_all.to_csv()
else:
   print df_all




