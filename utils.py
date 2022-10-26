# -*- coding: utf-8 -*-
"""
Helper functions

@author: tanner stevenson
"""

import numpy as np
import os


def is_scalar(x):
    return np.ndim(x) == 0


def check_make_dir(full_path):
    path_dir = os.path.dirname(full_path)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)


def flatten_dict_array(dict_array):
    ''' flattens a dictionary of lists '''
    return [i for v in dict_array.values() for i in v]


def convert_to_multiple(value, factor, round_up=True):
    if round_up:
        return factor * np.ceil(value/factor)
    else:
        return factor * np.floor(value/factor)


def stderr(x, axis=0, ignore_nan=True):
    ''' Calculates standard error '''

    x = np.array(x)

    if ignore_nan:
        std = np.nanstd(x, axis)
        n = np.sum(np.logical_not(np.isnan(x)), axis)
    else:
        std = np.std(x, axis)
        n = np.shape(x)[axis]

    se = std/np.sqrt(n)
    se[np.isinf(se)] = np.nan  # handle cases where n = 0
    return se
