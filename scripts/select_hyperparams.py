#!/usr/bin/env python

import pandas as pd
import numpy as np
pd.options.display.width = 150
pd.options.display.max_columns = 1000

def aggf(df):
    df = df.sort_values(by="val_f1", ascending=False)
    df = df.iloc[:5, :]
    df = df.sort_values(by="val_recall", ascending=False)
    return df.iloc[:1, :]

dfr = pd.read_csv('stats/results_gridsearch_811949.csv')
cols = filter(len, 'dirname kernel_sizes filters val_precision val_recall val_f1 val_aupr'.split())

print dfr.groupby('dataset').apply(aggf).loc[:, cols]
