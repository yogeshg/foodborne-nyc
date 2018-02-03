#!/usr/bin/env python

import pandas as pd
import numpy as np
import seaborn as sns
from itertools import product
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

pd.options.display.width = 150
pd.options.display.max_columns = 1000

def aggf(df):
    df = df.sort_values(by="val_f1", ascending=False)
    df = df.iloc[:5, :]
    df = df.sort_values(by="val_recall", ascending=False)
    return df.iloc[:1, :]

def get_hex(cm, x, a=0, b=1, c=0, d=1):
    x = c + (d-c)*(float(x) - a)/(b-a)
    return matplotlib.colors.rgb2hex(cm(x))

def test_get_hex():
    cm = LinearSegmentedColormap.from_list('black_to_white', ['black', 'white'])
    return [(x, get_hex(cm, x, 0, 16, 16/16., 15/16.)) for x in range(16)]


def heatmap_styler(x, **kwargs):
    return 'background-color: {}'.format(get_hex(cm, x, **kwargs))

def border_box_styler(df, border_style="1px solid #000000"):
    styles = df.copy()
    styles.loc[:,:] = ""
    styles.iloc[0,:] += " border-top: {};".format(border_style)
    styles.iloc[-1,:] += " border-bottom: {};".format(border_style)
    styles.iloc[:,0] += " border-left: {};".format(border_style)
    styles.iloc[:,-1] += " border-right: {};".format(border_style)
    return styles

import sys

usage = """
    {} csv_fname
"""
assert len(sys.argv) > 1, usage.format(sys.argv[0])

all_experiments = None
for csv_fname in sys.argv[1:]:
    dfr = pd.read_csv(csv_fname)
    dfr.dataset = dfr.dataset.map(lambda x: (' '.join(x.split('.')[1:])).title())
    dfi = dfr.set_index('dataset,kernel_sizes,filters'.split(','))
    if all_experiments is None:
        all_experiments = dfi
    all_experiments.update(dfi)

cols = filter(len, 'dirname kernel_sizes filters val_precision val_recall val_f1 val_aupr'.split())

print(all_experiments.sort_index().dirname.unstack())
all_experiments = all_experiments.sortlevel(level=[0,1], ascending=[True, False]).reset_index()
# raise ValueError, "BREAK"

groupby_per_regime = ['kernel_sizes', 'filters']
agg_per_regime = {'val_f1': 'first', 'val_recall': 'first', 'val_precision': 'first'}
df1 = all_experiments.groupby(['dataset']).apply(lambda df: df.groupby(groupby_per_regime).agg(agg_per_regime))\
        .unstack('kernel_sizes')
df1.to_csv('./stats/results-hyperparams-pivot.csv')
df3 = df1.loc[:, 'kernel_sizes,filters,val_precision,val_recall,val_f1'.split(',')]
df3.to_latex('./stats/results-hyperparams-pivot.tex')

cm = LinearSegmentedColormap.from_list('high_green', ['white', 'white', 'white', 'seagreen'])
styled = df1.style.format("{:.2%}")

for d,m in list(product(list(df1.index.levels[0]), list(df1.columns.levels[0]))):
    style_slice = pd.IndexSlice[(d,):(d,), (m,):(m,)]
    low = df1.loc[style_slice].min().min()
    high = df1.loc[style_slice].max().max()
    styled = styled.applymap(heatmap_styler, subset=style_slice, a=low, b=high, d=0.9)
    styled = styled.apply(border_box_styler, subset=style_slice, axis=None, border_style="2px solid #000000")


html = styled.render()
with open ('./stats/results-hyperparams-pivot.html', 'w') as f:
    f.write(html)



df2 = all_experiments.groupby('dataset').apply(aggf).loc[:, cols].reset_index()
df2.to_csv('./stats/results-best-hyperparams.csv')
df2 = df2.loc[:, 'dataset,kernel_sizes,filters,val_precision,val_recall,val_f1'.split(',')]
df2.to_latex('./stats/results-best-hyperparams.tex', float_format="%.4f", index=None)


