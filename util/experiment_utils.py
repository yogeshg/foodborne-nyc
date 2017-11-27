import pandas as pd
import json
import logging
logger = logging.getLogger(__name__)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_metric(df, metric_name, i, dirpath):
    assert type(df) == pd.DataFrame, type(df)
    assert type(metric_name) == str, type(metric_name)
    assert type(i) == int, i
    assert type(dirpath) == str, dirpath
    val_metric = 'val_{}'.format(metric_name)
    cname = 'val_{}_{:04d}'.format(metric_name, i)
    df.loc[:, cname] = df.loc[i, val_metric]
    df.loc[:, [metric_name, val_metric, cname]].plot()
    plt.savefig(dirpath + '/{}.png'.format(metric_name))
    return

