"""LSTM-based encoders and decoders for MusicVAE."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn as rnn_ops
from tensorflow.python.layers import convolutional
from tensorflow.python.layers import normalization
from tensorflow.python.layers import pooling
from tensorflow.python.framework import dtypes

from TFLibrary.Seq2Seq import base_models
from TFLibrary.Seq2Seq import rnn_cell_utils
from TFLibrary.utils import tensorflow_utils


class LstmEncoder(base_models.BaseEncoder):
    """Unidirectional LSTM Encoder."""

    def __init__(self,
                 unit_type,
                 num_units,
                 num_layers=1,
                 dropout_rate=None,
                 num_residual_layers=0,
                 scope="LstmEncoder",
                 is_training=True,  # only for dropout
                 bidirectional=True):
        
        self._encoder_scope = scope
        self._is_training = is_training
        self._bidirectional = bidirectional
        
        self._unit_type = unit_type
        self._num_units = num_units
        self._num_layers = num_layers
        self._dropout_rate = dropout_rate
        self._num_residual_layers = num_residual_layers

    def build(self):
        mode = "train" if self._is_training else "inference"

        if self._bidirectional:
            self._fw_cell = rnn_cell_utils.create_rnn_cell(
                unit_type=self._unit_type,
                num_units=self._num_units,
                num_layers=self._num_layers,
                mode=mode,
                dropout=self._dropout_rate,
                num_residual_layers=self._num_residual_layers,
                # use default cell creator
                single_cell_fn=None)

            self._bw_cell = rnn_cell_utils.create_rnn_cell(
                unit_type=self._unit_type,
                num_units=self._num_units,
                num_layers=self._num_layers,
                mode=mode,
                dropout=self._dropout_rate,
                num_residual_layers=self._num_residual_layers,
                # use default cell creator
                single_cell_fn=None)

        else:
            self._cell = rnn_cell_utils.create_rnn_cell(
                unit_type=self._unit_type,
                num_units=self._num_units,
                num_layers=self._num_layers,
                mode=mode,
                dropout=self._dropout_rate,
                num_residual_layers=self._num_residual_layers,
                # use default cell creator
                single_cell_fn=None)


    def encode(self, inputs, sequence_length=None, initial_state=None):
        if self._bidirectional:
            outputs, state = rnn_ops.bidirectional_dynamic_rnn(
                cell_fw=self._fw_cell,
                cell_bw=self._bw_cell,
                inputs=inputs,
                sequence_length=sequence_length,
                initial_state_fw=initial_state,
                initial_state_bw=initial_state,
                dtype=dtypes.float32,
                time_major=False,
                scope=self._encoder_scope)
            # concatenate the forwards and backwards states
            outputs = array_ops.concat(axis=2, values=outputs)
        else:
            outputs, state = rnn_ops.dynamic_rnn(
                cell=self._cell,
                inputs=inputs,
                sequence_length=sequence_length,
                initial_state=initial_state,
                dtype=dtypes.float32,
                time_major=False,
                scope=self._encoder_scope)
        
        return outputs, state


class TempConvEncoder(base_models.BaseEncoder):
    def __init__(self,
                 filters,
                 kernel_sizes,
                 pool_size=2,
                 strides=1,
                 padding='same',
                 activation=nn_ops.relu,
                 use_bias=True,
                 scope="TempConvEncoder",
                 is_training=True):
        if not isinstance(kernel_sizes, (tuple, list)):
            raise ValueError("kernel_sizes must be a list")

        self._filters = filters
        self._kernel_sizes = kernel_sizes
        self._pool_size = pool_size
        self._strides = strides
        self._padding = padding
        self._activation = activation
        self._use_bias = use_bias
        
        self._encoder_scope = scope
        self._is_training = is_training

    def build(self):
        norm_layers = []
        conv_layers = []
        pool_layers = []

        for layer_id, kernel_size in enumerate(self._kernel_sizes):
            norm_layer = normalization.BatchNormalization(
                name="BatchNormalization_%d" % layer_id)
            conv_layer = convolutional.Conv1D(
                filters=self._filters,
                kernel_size=kernel_size,
                strides=self._strides,
                padding=self._padding,
                activation=self._activation,
                use_bias=self._use_bias,
                name="Conv1D_%d" % layer_id)
            pool_layer = pooling.MaxPooling1D(
                pool_size=self._pool_size,
                strides=self._strides,
                padding=self._padding,
                name="MaxPooling1D_%d" % layer_id)

            norm_layers.append(norm_layer)
            conv_layers.append(conv_layer)
            pool_layers.append(pool_layer)

        self._norm_layers = norm_layers
        self._conv_layers = conv_layers
        self._pool_layers = pool_layers

    def encode(self, inputs):
        all_outputs = []
        for norm_layer, conv_layer, pool_layer in zip(
                self._norm_layers, self._conv_layers, self._pool_layers):

            outputs = norm_layer(inputs, training=self._is_training)
            outputs = conv_layer(outputs)
            outputs = pool_layer(outputs)
            all_outputs.append(outputs)

        all_outputs = math_ops.reduce_mean(all_outputs, axis=-1)
        return outputs
