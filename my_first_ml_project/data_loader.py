from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import numpy as np
import tensorflow as tf
import pandas as pd
import scipy as sp
import sklearn.preprocessing as skp

from constants import FEATURE_COLUMNS, CATEGORICAL_FEATURE_KEYS


def load_data(path, data_type):
    data = pd.read_csv(path, index_col="Instance")
    if data_type == "train":
        data.dropna(inplace=True)

    Y = data.pop('Income')

    for key in CATEGORICAL_FEATURE_KEYS:
        data[key] = pd.Categorical(data[key])
        data[key] = data[key].cat.codes

    data = pd.DataFrame(skp.scale(data), columns=FEATURE_COLUMNS)

    dataset = tf.data.Dataset.from_tensor_slices((data.values, Y.values))
    for feat, targ in dataset.take(5):
        print('Features: {}, Target: {}'.format(feat, targ))

    dataset = tf.data.Dataset.from_tensor_slices((data.values, Y.values))

    return dataset
