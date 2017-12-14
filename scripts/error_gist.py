import logging

import numpy as np
import pandas as pd
from numpy import array, float32

from datasets.experiments.baseline_experiment_util import importance_weighted_precision_recall

logging.basicConfig(level=logging.INFO)
pd.options.display.width = 180


def update(df):
    df.loc[:, 'y_predicted'] = (df.y_predicted > 0.5).astype(int)
    df.loc[:, 'true_positive'] = ((df.y_predicted.astype(bool)) & (df.y_true.astype(bool))).astype(int) * weights
    df.loc[:, 'false_positive'] = ((df.y_predicted.astype(bool)) & (~(df.y_true.astype(bool)))).astype(int) * weights
    df.loc[:, 'false_negative'] = ((~(df.y_predicted.astype(bool))) & (df.y_true.astype(bool))).astype(int) * weights
    df.loc[:, 'true_negative'] = ((~(df.y_predicted.astype(bool))) & (~(df.y_true.astype(bool)))).astype(int) * weights

    return df


def test(y_true, y_predicted, is_biased, weights):
    df = pd.DataFrame({'y_true': y_true, 'y_predicted': y_predicted, 'is_biased': is_biased, 'weights': weights})
    print df

    # update(df)
    # print df

    _p, _r = importance_weighted_precision_recall(y_true, y_predicted, is_biased)
    print("experiment util precision: "+str(_p))
    print("experiment util recall: "+str(_r))


# p1, p2, (y_true, y_predicted, is_biased, weights) = e.args[0]

print('test case 1')
(y_true, y_predicted, is_biased, weights) = \
        (array([0, 0, 0, 0, 0, 1, 1, 1]),
         array([ 0.15469682,  0.23727584,  0.35687983,  0.11834912,  0.34073523,
                 0.17484501,  0.15119889,  0.17018321], dtype=float32),
         array([1, 1, 1, 1, 1, 1, 1, 1]),
         array([ 0.20770626,  0.20770626,  0.20770626,  0.20770626,  0.20770626,
                 0.20770626,  0.20770626,  0.20770626]))
# test(y_true, y_predicted, is_biased, weights)


print('test case 2')
is_biased = np.array([1, 1, 1, 1, 1, 1, 1, 0])
y_true = np.array([0, 1, 0, 0, 1, 1, 1, 1])
weights = np.array([0.207706, 0.207706, 0.207706, 0.207706, 0.207706, 0.207706, 0.207706, 7.485170])
y_predicted = np.array([-1.775076,   0.879030,  -1.643957,  -0.384973,   0.491523,   0.655548,   0.800963,  -1.677646])
# test(y_true, y_predicted, is_biased, weights)


print('test case 3')
n = 20
np.random.seed(1991)
is_biased = np.array([1 for _ in range(n//2)] + [0 for _ in range(n//2)])
y_true = (np.random.uniform(0, 1, (n,)) > 0.5)*1.0
weights = is_biased * 12.3 + 0.456
y_predicted = np.random.uniform(0, 1, (n,))
test(y_true, y_predicted, is_biased, weights)
