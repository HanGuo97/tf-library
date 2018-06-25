"""Transformer Utilities

https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from TFLibrary.utils import scope_utils
from tensor2tensor.layers import common_layers


# ==========================================
# Layers
# ==========================================

def embedding(*args, **kwargs):
    """I'm Lazy"""
    return common_layers.embedding(*args, **kwargs)


def dense_relu_dense(inputs,
                     filter_size,
                     output_size,
                     dropout_rate=0.0,
                     name=None):
    """Linear Prjection + RELU + Linear Prjection"""
    layer_name = "%s_{}" % name if name else "{}"
    h = tf.layers.dense(
        inputs, filter_size,
        use_bias=True,
        activation=tf.nn.relu,
        name=layer_name.format("conv1"))

    if dropout_rate != 0.0:
        h = tf.nn.dropout(h, keep_prob=1.0 - dropout_rate)

    o = tf.layers.dense(
        h, output_size,
        activation=None,
        use_bias=True,
        name=layer_name.format("conv2"))

    return o


def residual_connection(x, previous_value):
    return x + previous_value


def dropout(x, dropout_rate):
    return tf.nn.dropout(x, keep_prob=1.0 - dropout_rate)


def _layer_norm(x, epsilon, scale, bias):
    """Layer norm raw computation."""
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * scale + bias


def layer_norm(x, filters=None, epsilon=1e-6, name=None, reuse=None):
    """Layer normalize the tensor x, averaging over the last dimension."""
    if filters is None:
        filters = shape_list(x)[-1]
    with tf.variable_scope(
            name, default_name="layer_norm", values=[x], reuse=reuse):
        scale = tf.get_variable(
            "layer_norm_scale", [filters],
            initializer=tf.ones_initializer())
        bias = tf.get_variable(
            "layer_norm_bias", [filters],
            initializer=tf.zeros_initializer())

        return _layer_norm(x, epsilon, scale, bias)


# ==========================================
# Positional Embedding
# ==========================================

@scope_utils.add_name_scope()
def add_positional_embedding(x, max_length, name):
    """Add positional embedding.
    Args:
        x: a Tensor with shape [batch, length, depth]
        max_length: an integer.  static maximum size of any dimension.
        name: a name for this layer.
    Returns:
        a Tensor the same shape as x.
    """
    _, length, depth = shape_list(x)
    var = tf.get_variable(name, [max_length, depth])
    return x + tf.expand_dims(tf.slice(var, [0, 0], [length, -1]), 0)


# ==========================================
# MultiHead Simplitng and Merging
# ==========================================

def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


@scope_utils.add_name_scope()
def combine_last_two_dimensions(x):
    """Reshape x so that the last two dimension become one.
    Args:
      x: a Tensor with shape [..., a, b]
    Returns:
      a Tensor with shape [..., ab]
    """
    x_shape = shape_list(x)
    a, b = x_shape[-2:]
    return tf.reshape(x, x_shape[:-2] + [a * b])


@scope_utils.add_name_scope()
def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.
    The first of these two dimensions is n.
    Args:
        x: a Tensor with shape [..., m]
        n: an integer.
    Returns:
        a Tensor with shape [..., n, m/n]
    """
    x_shape = shape_list(x)
    m = x_shape[-1]
    if isinstance(m, int) and isinstance(n, int):
        assert m % n == 0
    return tf.reshape(x, x_shape[:-1] + [n, m // n])


@scope_utils.add_name_scope()
def combine_heads(x):
    """Inverse of split_heads.
    Args:
        x: a Tensor with shape [batch, num_heads, length, channels / num_heads]
    Returns:
        a Tensor with shape [batch, length, channels]
    """
    return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))


@scope_utils.add_name_scope()
def split_heads(x, num_heads):
    """Split channels (dimension 2) into multiple heads (becomes dimension 1).
    Args:
        x: a Tensor with shape [batch, length, channels]
        num_heads: an integer
    Returns:
        a Tensor with shape [batch, num_heads, length, channels / num_heads]
    """
    return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])


# ==========================================
# MultiHead Attention
# ==========================================

def compute_qkv(query_antecedent,
                memory_antecedent,
                total_key_depth,
                total_value_depth):
    """Computes query, key and value.
    Args:
        query_antecedent: [batch, length_q, channels]
        memory_antecedent: [batch, length_m, channels]
        total_key_depth: an integer
        total_value_depth: an integer
    Returns:
        q, k, v : [batch, length, depth] tensors
    """
    if memory_antecedent is None:
        memory_antecedent = query_antecedent
    q = tf.layers.dense(
        query_antecedent, total_key_depth,
        use_bias=False, name="q")
    k = tf.layers.dense(
        memory_antecedent, total_key_depth,
        use_bias=False, name="k")
    v = tf.layers.dense(
        memory_antecedent, total_value_depth,
        use_bias=False, name="v")

    return q, k, v


def dot_product_attention(q, k, v, dropout_rate=0.0, name=None):
    """dot-product attention.
    Args:
        q: [batch, heads, length_q, depth_k]
        k: [batch, heads, length_kv, depth_k]
        v: [batch, heads, length_kv, depth_v]
        dropout_rate: a floating point number
        name: an optional string
    Returns:
        A Tensor.
    """
    with tf.variable_scope(name,
                           default_name="dot_product_attention",
                           values=[q, k, v]) as scope:
        # [batch, num_heads, query_length, memory_length]
        logits = tf.matmul(q, k, transpose_b=True)
        weights = tf.nn.softmax(logits, name="attention_weights")
        # dropping out the attention links for each of the heads
        weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
        return tf.matmul(weights, v)


def multihead_attention(query_antecedent,
                        memory_antecedent,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        name="multihead_attention"):
    """Multihead scaled-dot-product attention with input/output transformations.
    Args:
      query_antecedent: a Tensor with shape [batch, length_q, channels]

      total_key_depth: an integer
      total_value_depth: an integer
      output_depth: an integer
      num_heads: an integer dividing total_key_depth and total_value_depth
      dropout_rate: a floating point number
      name: an optional string.

    Returns:
        The result of the attention transformation.
            The output shape is [batch_size, length_q, hidden_dim]

    Raises:
        ValueError: if the key depth or value depth are not divisible by the
            number of attention heads.
    """
    if total_key_depth % num_heads != 0:
        raise ValueError(
            "Key depth (%d) must be divisible by the number of "
            "attention heads (%d)." % (total_key_depth, num_heads))
    if total_value_depth % num_heads != 0:
        raise ValueError(
            "Value depth (%d) must be divisible by the number of "
            "attention heads (%d)." % (total_value_depth, num_heads))

    with tf.variable_scope(name,
                           default_name="multihead_attention",
                           values=[query_antecedent, memory_antecedent]):

        q, k, v = compute_qkv(
            query_antecedent=query_antecedent,
            memory_antecedent=memory_antecedent,
            total_key_depth=total_key_depth,
            total_value_depth=total_value_depth)

        q = split_heads(q, num_heads)
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)

        key_depth_per_head = total_key_depth // num_heads
        q *= key_depth_per_head**-0.5
        x = dot_product_attention(q, k, v, dropout_rate=dropout_rate)
        x = combine_heads(x)
        # Set last dim specifically.
        x.set_shape(x.shape.as_list()[:-1] + [total_value_depth])
        x = tf.layers.dense(
            x,
            units=output_depth,
            use_bias=False,
            name="output_transform")

        return x
