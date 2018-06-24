"""Utilities for Tensorflow Scopes

https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import functools
import tensorflow as tf


def add_scope(scope=None, scope_fn=None):
    """Return a decorator which add a TF name/variable scope to a function.
    Note that the function returned by the decorator accept additional 'name'
    parameter, which can overwrite the name scope given when the function is
    created.
    Args:
        scope (str): name of the scope. If None, the function name is used.
        scope_fn (fct): Either tf.name_scope or tf.variable_scope
    Returns:
        fct: the add_scope decorator
    """
    def decorator(f):

        @functools.wraps(f)
        def decorated(*args, **kwargs):
            # Python 2 hack for keyword only args
            name = kwargs.pop("name", None)
            with scope_fn(name or scope or f.__name__):
                return f(*args, **kwargs)

        return decorated

    return decorator


def add_var_scope(scope=None):
    return add_scope(scope, scope_fn=tf.variable_scope)


def add_name_scope(scope=None):
    return add_scope(scope, scope_fn=tf.name_scope)
