# Copyright 2017 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Utility functions for dealing with Sonnet Modules.
https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/util.py"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import six
import math
import tensorflow as tf


def get_variable_scope_name(value):
    """Returns the name of the variable scope indicated by the given value.

    Args:
      value: String, variable scope, or object with `variable_scope` attribute
      (e.g., Sonnet module).

    Returns:
      The name (a string) of the corresponding variable scope.

    Raises:
      ValueError: If `value` does not identify a variable scope.
    """
    # If the object has a "variable_scope" property, use it.
    value = getattr(value, "variable_scope", value)
    if isinstance(value, tf.VariableScope):
        return value.name
    elif isinstance(value, six.string_types):
        return value
    else:
        raise ValueError("Not a variable scope: {}".format(value))


def get_variables_in_scope(scope, collection=tf.GraphKeys.TRAINABLE_VARIABLES):
    """Returns a tuple `tf.Variable`s in a scope for a given collection.

    Args:
      scope: `tf.VariableScope` or string to retrieve variables from.
      collection: Collection to restrict query to. By default this is
          `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include
          non-trainable variables such as moving averages.

    Returns:
      A tuple of `tf.Variable` objects.
    """
    scope_name = get_variable_scope_name(scope)

    if scope_name:
        # Escape the name in case it contains any "." characters. Add a closing
        # slash so we will not search any scopes that have this scope name as a
        # prefix.
        scope_name = re.escape(scope_name) + "/"

    return tuple(tf.get_collection(collection, scope_name))


def get_variables_in_module(module,
                            collection=tf.GraphKeys.TRAINABLE_VARIABLES):
    """Returns tuple of `tf.Variable`s declared inside an `snt.Module`.

    Note that this operates by searching the variable scope a module contains,
    and so does not know about any modules which were constructed elsewhere but
    used inside this module.

    Args:
      module: `snt.Module` instance to query the scope of.
      collection: Collection to restrict query to. By default this is
        `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
        variables such as moving averages.

    Returns:
      A tuple of `tf.Variable` objects.

    Raises:
      NotConnectedError: If the module is not connected to the Graph.
    """
    return module.get_variables(collection=collection)


def create_linear_initializer(input_size, dtype=tf.float32):
    """Returns a default initializer for weights of a linear module."""
    stddev = 1 / math.sqrt(input_size)
    return tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)


def create_bias_initializer(unused_bias_shape, dtype=tf.float32):
    """Returns a default initializer for the biases of linear module."""
    return tf.zeros_initializer(dtype=dtype)
