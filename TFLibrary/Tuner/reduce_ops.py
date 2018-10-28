import numpy as np


def no_op(values):
    return None


def reduce_min(values):
    return np.min(values)


def reduce_mean(values):
    return np.mean(values)
