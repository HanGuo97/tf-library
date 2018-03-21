"""LSTM-based encoders and decoders for MusicVAE."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn as rnn_ops
from tensorflow.python.framework import dtypes

from TFLibrary.Seq2Seq import base_model
from TFLibrary.Seq2Seq import rnn_cell_utils


class LstmEncoder(base_model.BaseEncoder):
    """Unidirectional LSTM Encoder."""

    def __init__(self,
                 hparams,
                 scope="encoder",
                 is_training=True,
                 bidirectional=True):

        if not isinstance(hparams, base_model.EncoderHParams):
            raise TypeError("Expected hparams to be EncoderHParams")
        
        self._hparams = hparams
        self._encoder_scope = scope
        self._is_training = is_training
        self._bidirectional = bidirectional


    @property
    def output_size(self):
        return self._cell.output_size

    def build(self):
        hps = self._hparams
        mode = "train" if self._is_training else "inference"

        if self._bidirectional:
            self._fw_cell = rnn_cell_utils.create_rnn_cell(
                unit_type=hps.unit_type,
                num_units=hps.num_units,
                num_layers=hps.num_layers,
                mode=mode,
                dropout=hps.dropout_rate,
                num_residual_layers=hps.num_residual_layers,
                # use default cell creator
                single_cell_fn=None)

            self._bw_cell = rnn_cell_utils.create_rnn_cell(
                unit_type=hps.unit_type,
                num_units=hps.num_units,
                num_layers=hps.num_layers,
                mode=mode,
                dropout=hps.dropout_rate,
                num_residual_layers=hps.num_residual_layers,
                # use default cell creator
                single_cell_fn=None)

        else:
            self._cell = rnn_cell_utils.create_rnn_cell(
                unit_type=hps.unit_type,
                num_units=hps.num_units,
                num_layers=hps.num_layers,
                mode=mode,
                dropout=hps.dropout_rate,
                num_residual_layers=hps.num_residual_layers,
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
