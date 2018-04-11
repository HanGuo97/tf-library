"""LSTM-based encoders and decoders for MusicVAE."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn as rnn_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.layers import convolutional
from tensorflow.python.layers import normalization
from tensorflow.python.layers import pooling

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import ops as framework_ops

from tensorflow.python.platform import tf_logging as logging_ops

from TFLibrary.Seq2Seq import base_models
from TFLibrary.Seq2Seq import rnn_cell_utils
from TFLibrary.Seq2Seq import lstm_utils
from TFLibrary.utils import tensorflow_utils

import numpy as np
from collections import namedtuple
HierEncTuple = namedtuple("HierEncTuple", ("NumSplits", "Encoder", "Scope"))


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
        logging_ops.warn("Scope is actually not used!!!")

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

        # [batch_size, sequence_length, num_filters x num_kernels]
        all_outputs = array_ops.concat(all_outputs, axis=-1)
        # [batch_size, num_filters x num_kernels]
        all_outputs = math_ops.reduce_mean(all_outputs, axis=1)
        return all_outputs


class HierarchicalLstmEncoder(base_models.BaseEncoder):
    """Hierarchical LSTM encoder wrapper.
    Input sequences will be split into segments based on the first value of
    `level_lengths` and encoded. At subsequent levels, the embeddings will be
    grouped based on `level_lengths` and encoded until a single embedding is
    produced.
    See the `encode` method for details on the expected arrangement the sequence
    tensors.
    Args:
      core_encoder_cls: A single BaseEncoder class to use for each level of the
        hierarchy.
      level_lengths: A list of the (maximum) lengths of the segments at each
        level of the hierarchy. The product must equal `hparams.max_seq_len`.
    """

    def __init__(self, core_encoder_cls, level_lengths, total_length, **kargs):

        if not issubclass(core_encoder_cls, base_models.BaseEncoder):
            raise TypeError(
                "Expected `core_encoder_cls` to be "
                "a subclass of base_models.BaseEncoder")

        if not isinstance(level_lengths, (tuple, list)):
            raise TypeError(
                "Expected `level_lengths` to be tuple or list "
                "found ", type(level_lengths))

        # the product of level lengths must equal total lengths
        # a.k.a. maximum sequence lengths
        if total_length != np.prod(level_lengths):
            raise ValueError(
                "The product of the HierarchicalLstmEncoder "
                "level lengths (%d) must equal the padded "
                "input sequence length (%d)." % (
                    np.prod(level_lengths), total_length))

        self._core_encoder_cls = core_encoder_cls
        self._level_lengths = level_lengths
        self._total_length = total_length

        # for building the model
        self._kargs = kargs

    @property
    def level_lengths(self):
        return list(self._level_lengths)

    def level(self, l):
        """Returns the BaseEncoder at level `l`."""
        return self._hierarchical_encoders[l].Encoder

    def build(self):
        logging_ops.info(
            "\nHierarchical Encoder:\n"
            "  input length: %d\n"
            "  level lengths: %s\n",
            self._total_length,
            self._level_lengths)

        hierarchical_encoders = []
        num_splits = np.prod(self._level_lengths)
        for i, l in enumerate(self._level_lengths):
            num_splits //= l
            
            with vs.variable_scope("hierarchical_encoder/level_%d" % i,
                                   reuse=vs.AUTO_REUSE) as scope:
                
                h_encoder = self._core_encoder_cls(scope=scope, **self._kargs)
                h_encoder.build()

            hierarchical_encoders.append(
                HierEncTuple(NumSplits=num_splits,
                             Encoder=h_encoder,
                             Scope=scope))

            logging_ops.info(
                "Level %d \tsplits: %d \tlength %d", i, num_splits, l)

        self._hierarchical_encoders = hierarchical_encoders

    def encode(self, sequence, sequence_length):
        """Hierarchically encodes the input sequences, returning a single embedding.
        Each sequence should be padded per-segment. For example, a sequence with
        three segments [1, 2, 3], [4, 5], [6, 7, 8 ,9] and a `max_seq_len` of 12
        should be input as `sequence = [1, 2, 3, 0, 4, 5, 0, 0, 6, 7, 8, 9]` with
        `sequence_length = [3, 2, 4]`.
        Args:
          sequence: A batch of (padded) sequences, sized
            `[batch_size, max_seq_len, input_depth]`.
          sequence_length: A batch of sequence lengths. May be sized
            `[batch_size, level_lengths[0]]` or `[batch_size]`. If the latter,
            each length must either equal `max_seq_len` or 0. In this case, the
            segment lengths are assumed to be constant and the total length will be
            evenly divided amongst the segments.
        Returns:
          embedding: A batch of embeddings, sized `[batch_size, N]`.
        """
        if not tensor_util.is_tensor(sequence):
            raise TypeError("`sequence` must be tf.Tensor")
        if not tensor_util.is_tensor(sequence_length):
            raise TypeError("`sequence_length` must be tf.Tensor")

        batch_size = sequence.shape[0].value

        # suppose sequence length = [100] * batch_size,
        # num_splits = 5, then the sequence_length will be
        # [20, 20, 20, 20, 20] * batch_size

        # suppose level_lengths = [12, 6, 2, 1]
        # this function will produce num_splits = 6 x 2 x 1 = 12
        sequence_length = lstm_utils.maybe_split_sequence_lengths(
            sequence_length=sequence_length,
            num_splits=np.prod(self._level_lengths[1:]),
            total_length=self._total_length)

        for level, (num_splits, h_encoder, scope) in enumerate(
                self._hierarchical_encoders):
            # encoder.encode()
            # takes two arguments: sequence, and sequence_length

            # Compute Split sequences
            # split sequences according to level_lengths[level]
            split_seqs = array_ops.split(sequence, num_splits, axis=1)

            # Compute Split sequence_length
            # In Level 0, we use the input `sequence_lengths`.
            # After that, we use the full embedding sequences.
            if level == 0:
                # Actually, I believe the level != 0 assignment
                # can still work on level == 0
                sequence_length = sequence_length
            else:
                # [single_seq_len, ...] for length num_splits
                # and tile over batch sizes
                single_seq_len = split_seqs[0].shape[1]
                sequence_length = array_ops.fill(
                    value=single_seq_len,
                    dims=[batch_size, num_splits])

            # from [batch_size, num_splits, num_units] to
            # list of [batch_size, num_units] with length num_splits
            split_lengths = array_ops.unstack(sequence_length, axis=1)
            
            # list of cell outputs and cell states with length num_splits
            outputs, states = zip(*[h_encoder.encode(s, l)
                for s, l in zip(split_seqs, split_lengths)])

            # to propagate into the next level
            # we only need the last cell states from the current level
            
            # we first extract the last h for each sub-sequence
            # concatenate them if there are multiple h (e.g. bidirectional)
            # then we stack them into the sequence_length dimention

            # here we hard-code using bidirectional LSTM, and thus
            last_states = lstm_utils.extract_and_concat_bidir_last_h(states)

            # back to [batch_size, num_splits, num_units]
            sequence = array_ops.stack(last_states, axis=1)


        # the last level must have sequence length 1
        with framework_ops.control_dependencies(
                [check_ops.assert_equal(
                    array_ops.shape(sequence)[1], 1,
                    message="Outputs of the last layer in "
                            "hierarchical encoders as "
                            "sequence_length == level_lengths[-1]")]):

            if len(outputs) != 1:
                raise ValueError("outputs in the last level must be 1")

            # we return both cell_outputs and last_cell_states
            # cell_outputs are the cell_outputs from the second last level
            # (since the last level) will have sequence_length 1
            # last_cell_states are just the sequence at the last level
            
            cell_outputs = outputs[0]
            last_cell_states = states[0]

            return cell_outputs, last_cell_states
