from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import numpy as np
import tensorflow as tf
import pandas as pd
import scipy as sp
import sklearn.preprocessing as skp

from constants import FEATURE_COLUMNS, CATEGORICAL_FEATURE_KEYS, NUMERIC_FEATURE_KEYS


def load_data():
    data_set = dict()

    train_data = pd.read_csv("data/tcd ml 2019-20 income prediction training (with labels).csv", index_col="Instance")
    train_data = train_data.fillna(train_data.mean()[NUMERIC_FEATURE_KEYS])

    for key in CATEGORICAL_FEATURE_KEYS:
        train_data[key] = pd.Categorical(train_data[key])
        train_data[key] = train_data[key].cat.codes

    train_data["Size of City"] = skp.scale(train_data["Size of City"])

    # generate dev and test sets
    dev_set = train_data.sample(11199)
    dev_index = dev_set.index
    for index in dev_index:
        train_data.drop(index, inplace=True)

    test_set = train_data.sample(11199)
    test_index = test_set.index
    for index in test_index:
        train_data.drop(index, inplace=True)

    data_set["train_Y"] = train_data.pop("Income")
    data_set["train_X"] = train_data

    data_set["dev_Y"] = dev_set.pop("Income")
    data_set["dev_X"] = dev_set

    data_set["test_Y"] = test_set.pop("Income")
    data_set["test_X"] = test_set

    return data_set


def preprocessing(data_set):

    return data_set
