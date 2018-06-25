"""Transformer-based encoders and decoders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from TFLibrary.Modules import base
from TFLibrary.Modules import transformer_utils


class TransformerEncoder(base.AbstractModule):
    """Transformer, encoder only."""
    def __init__(self,
                 num_units,
                 filter_size,
                 num_layers,
                 num_heads,
                 dropout_rate,
                 max_length,
                 target_space,
                 target_space_size,
                 is_training=True,  # only for dropout
                 scope=None,  # currently not used
                 name="TransformerEncoder"):
        """Create TranformerEncoder Module

        Args:
            num_units"
                Integer: hidden dimension
            filter_size:
                Integer: intermediate hidden dimension of the
                    2-layer FC layer after multi-head attention
            num_layers:
                Integer: number of (Multihead + FC) blocks
            num_heads:
                Integer: number of heads in MultiheadAttention
            dropout_rate:
                Float: keep_prob = 1 - dropout_rate.
                Will be set to 0.0 if `is_training` is False
            max_length:
                Integer: The maximum length of the sequences.
                max_length is used to create the embedding variable
                ing positional embedding which assigns an vector
                to location. However, the actual assigned embedding
                is dynamic using tf.slice(Tensor, TensorLength)
            target_space:
                Integer: the index of target_space.
                This is used to assign an task-specific embedded
                vector via assigning inputs = inputs + task-embedding
            target_space_size:
                Integer: The maximum integer for target spaces.
                This is used to create the embedding of spaces.


        TODO:
            1. test _build() for Transformer and TransformerEncoder
            2. check with OpenAI's implementation
        """

        super(TransformerEncoder, self).__init__(name=name)

        self._num_units = num_units
        self._filter_size = filter_size
        self._num_heads = num_heads
        self._num_layers = num_layers
        self._dropout_rate = dropout_rate

        self._max_length = max_length
        self._target_space = target_space
        self._target_space_size = target_space_size

        self._encoder_scope = scope
        self._is_training = is_training

    def _build(self, inputs):
        dropout_rate = self._dropout_rate if self._is_training else 0.0

        encoder_input = transformer_prepare_encoder(
            inputs=inputs,
            max_length=self._max_length,
            target_space=self._target_space,
            target_space_size=self._target_space_size)

        encoder_input = tf.nn.dropout(
            encoder_input, keep_prob=1.0 - dropout_rate)

        encoder_output = transformer_encoder(
            inputs=encoder_input,
            num_units=self._num_units,
            filter_size=self._filter_size,
            num_heads=self._num_heads,
            num_layers=self._num_layers,
            dropout_rate=dropout_rate,
            name="TransformerEncoder")
        
        return encoder_output


def transformer_prepare_encoder(inputs,
                                max_length,
                                target_space,
                                target_space_size):
    """Prepare for the Transformer Encoder.

        1. Task-specific embedding
        2. Positional embedding
    """

    # Append target_space_id embedding to inputs.
    emb_target_space = transformer_utils.embedding(
        target_space,
        vocab_size=target_space_size,
        dense_size=inputs.shape.as_list()[-1],
        name="target_space_embedding",
        dtype=tf.float32)

    # expand two dimensions
    emb_target_space = tf.reshape(emb_target_space, [1, 1, -1])
    encoder_input = inputs + emb_target_space

    encoder_input = transformer_utils.add_positional_embedding(
        encoder_input, max_length,
        "inputs_positional_embedding")

    return encoder_input


def _apply_layer(inputs, layer_fn, dropout_rate, scope, **layer_fn_kwargs):
    """prepro + layer + postpro. dropout Rate is shared"""
    with tf.variable_scope(scope or layer_fn.__name__):
        processed_inputs = _layer_preprocess(inputs)

        outputs = layer_fn(
            processed_inputs,
            dropout_rate=dropout_rate,
            **layer_fn_kwargs)

        processed_outputs = _layer_postprocess(
            outputs,
            inputs=processed_inputs,
            dropout_rate=dropout_rate)

        return processed_outputs


def _layer_preprocess(inputs):
    """Layer Norm"""
    return transformer_utils.layer_norm(inputs)


def _layer_postprocess(outputs, inputs, dropout_rate):
    """Dropout + Residual"""
    return transformer_utils.residual_connection(
        transformer_utils.dropout(outputs, dropout_rate), inputs)


def transformer_encoder(inputs,
                        num_units,
                        filter_size,
                        num_heads,
                        num_layers,
                        dropout_rate,
                        name="encoder"):
    """A stack of transformer layers."""
    next_input = inputs
    with tf.variable_scope(name):
        for layer in range(num_layers):
            with tf.variable_scope("layer_%d" % layer):
                # self attention
                next_input = _apply_layer(
                    inputs=next_input,
                    layer_fn=transformer_utils.multihead_attention,
                    dropout_rate=dropout_rate,
                    scope="self_attention",
                    # multihead_attention kwargs
                    memory_antecedent=None,
                    total_key_depth=num_units,
                    total_value_depth=num_units,
                    output_depth=num_units,
                    num_heads=num_heads)

                # linear projection
                next_input = _apply_layer(
                    inputs=next_input,
                    layer_fn=transformer_utils.dense_relu_dense,
                    dropout_rate=dropout_rate,
                    scope="ffn",
                    # dense_relu_dense kwargs
                    filter_size=filter_size,
                    output_size=num_units)

        # if normalization is done in layer_preprocess,
        # then it should also be done on the output,
        # since the output can grow very large, being
        # the sum of a whole stack of unnormalized layer outputs.
        return _layer_preprocess(next_input)
