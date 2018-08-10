"""LSTM-based encoders and decoders."""

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

from TFLibrary.Modules import base
from TFLibrary.Seq2Seq import rnn_cell_utils
from TFLibrary.Seq2Seq import lstm_utils
from TFLibrary.utils import tensorflow_utils
from TFLibrary.utils import misc_utils


class LstmEncoder(base.AbstractModule):
    """Unidirectional LSTM Encoder."""

    def __init__(self,
                 unit_type,
                 num_units,
                 num_layers=1,
                 dropout_rate=None,
                 num_residual_layers=0,
                 scope="LstmEncoder",
                 is_training=True,  # only for dropout
                 bidirectional=True,
                 name="LstmEncoder",
                 **encoder_kargs):

        super(LstmEncoder, self).__init__(name=name)

        self._encoder_scope = scope
        self._is_training = is_training
        self._bidirectional = bidirectional

        self._unit_type = unit_type
        self._num_units = num_units
        self._num_layers = num_layers
        self._dropout_rate = dropout_rate
        self._num_residual_layers = num_residual_layers

        self._encoder_kargs = encoder_kargs
        
        if encoder_kargs:
            print("Additional RNN Cell Arguments: \n")
            addi_info = ["\t\t\t %s \t %s " % (k, v)
                         for k, v in encoder_kargs.items()]
            print("\n".join(addi_info))
            print("\n")


    def _build(self, inputs, sequence_length=None, initial_state=None):
        mode = "train" if self._is_training else "inference"

        if self._bidirectional:
            fw_cell = rnn_cell_utils.create_rnn_cell(
                unit_type=self._unit_type,
                num_units=self._num_units,
                num_layers=self._num_layers,
                mode=mode,
                dropout=self._dropout_rate,
                num_residual_layers=self._num_residual_layers,
                # use default cell creator
                single_cell_fn=None,
                **self._encoder_kargs)

            bw_cell = rnn_cell_utils.create_rnn_cell(
                unit_type=self._unit_type,
                num_units=self._num_units,
                num_layers=self._num_layers,
                mode=mode,
                dropout=self._dropout_rate,
                num_residual_layers=self._num_residual_layers,
                # use default cell creator
                single_cell_fn=None,
                **self._encoder_kargs)

            outputs, state = rnn_ops.bidirectional_dynamic_rnn(
                cell_fw=fw_cell,
                cell_bw=bw_cell,
                inputs=inputs,
                sequence_length=sequence_length,
                initial_state_fw=initial_state,
                initial_state_bw=initial_state,
                dtype=dtypes.float32,
                time_major=False,
                scope=self._encoder_scope)
            
            # concatenate the forwards and backwards states
            outputs = array_ops.concat(axis=2, values=outputs)

            self._cell = [fw_cell, bw_cell]

        else:
            cell = rnn_cell_utils.create_rnn_cell(
                unit_type=self._unit_type,
                num_units=self._num_units,
                num_layers=self._num_layers,
                mode=mode,
                dropout=self._dropout_rate,
                num_residual_layers=self._num_residual_layers,
                # use default cell creator
                single_cell_fn=None,
                **self._encoder_kargs)

            outputs, state = rnn_ops.dynamic_rnn(
                cell=cell,
                inputs=inputs,
                sequence_length=sequence_length,
                initial_state=initial_state,
                dtype=dtypes.float32,
                time_major=False,
                scope=self._encoder_scope)

            self._cell = cell

        return outputs, state

    def _clone(self, name):
        return type(self)(unit_type=self._unit_type,
                          num_units=self._num_units,
                          num_layers=self._num_layers,
                          dropout_rate=self._dropout_rate,
                          num_residual_layers=self._num_residual_layers,
                          scope=name,
                          is_training=self._is_training,
                          bidirectional=self._bidirectional,
                          name=name,
                          **self._encoder_kargs)
