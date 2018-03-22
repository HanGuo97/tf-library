# Copyright (c) 2017 Ben Poole & Friedemann Zenke
# MIT License -- see LICENSE for details
#
# This file is part of the code to reproduce the core results of:
# Zenke, F., Poole, B., and Ganguli, S. (2017). Continual Learning Through
# Synaptic Intelligence. In Proceedings of the 34th International Conference on
# Machine Learning, D. Precup, and Y.W. Teh, eds. (International Convention
# Centre, Sydney, Australia: PMLR), pp. 3987-3995.
# http://proceedings.mlr.press/v70/zenke17a.html
#
from pathint.utils import ema
from pathint.regularizers import quadratic_regularizer
from pathint.keras_utils import compute_fishers
import tensorflow as tf

"""
https://github.com/ganguli-lab/pathint/blob/master/pathint/protocols.py
A protocol is a function that takes as input some parameters and returns a tuple:
    (protocol_name, optimizer_kwargs)
The protocol name is just a string that describes the protocol.
The optimizer_kwargs is a dictionary that will get passed to KOOptimizer. It typically contains:
    step_updates, task_updates, task_metrics, regularizer_fn
"""


PATH_INT_PROTOCOL = lambda omega_decay, xi: (
    
    {
        'init_updates':  [
            ('cweights', lambda vars, w, prev_val: w.value()),
        ],
        'step_updates':  [
            ('grads2', lambda vars, w, prev_val: prev_val -
             vars['unreg_grads'][w] * vars['deltas'][w]),
        ],
        'task_updates':  [
            ('omega', lambda vars, w, prev_val: tf.nn.relu(ema(omega_decay, prev_val, vars[
             'grads2'][w] / ((vars['cweights'][w] - w.value())**2 + xi)))),
            #('cached_grads2', lambda vars, w, prev_val: vars['grads2'][w]),
            #('cached_cweights', lambda vars, w, prev_val: vars['cweights'][w]),
            ('cweights', lambda opt, w, prev_val: w.value()),
            ('grads2', lambda vars, w, prev_val: prev_val * 0.0),
        ],
        'regularizer_fn': quadratic_regularizer,
    })


def PathIntegralProtocal(object):
    def __init__(self, omega, xi):
        self._omega = omega
        self._xi = xi

    def __repr__(self):
        "path_int[omega=%s,xi=%s]" % (self._omega, self._xi)


def sum_regularizer_fn(weights, vars):
    reg = 0.0
    for w in weights:
        reg += tf.reduce_sum(vars['sum_omega'][w] * w**2
                             - 2 * vars['sum_omega_cweights'][w] * w
                             + vars['sum_omega_cweights_squared'][w])

    return reg
