"""LSTM-based encoders and decoders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import rnn as rnn_ops

from TFLibrary.Modules import base
from TFLibrary.Modules import attentions
from TFLibrary.Seq2Seq import rnn_cell_utils


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


class BiDAFStyleEncoder(base.AbstractModule):
    """BiDAF Style Encoder"""
    def __init__(self,
                 unit_type,
                 num_units,
                 num_layers=1,
                 dropout_rate=None,
                 num_residual_layers=0,
                 scope="BiDAFStyleEncoder",
                 is_training=True,  # only for dropout
                 bidirectional=True,
                 name="BiDAFStyleEncoder",
                 **encoder_kargs):

        super(BiDAFStyleEncoder, self).__init__(name=name)
        if not bidirectional:
            raise NotImplementedError(
                "Currently only Bidiretional LSTM is supported")

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

    def _build(self,
               sequence_1,
               sequence_2,
               sequence_1_length,
               sequence_2_length):

        # Build modules
        phrase_layer = LstmEncoder(
            unit_type=self._unit_type,
            num_units=self._num_units,
            num_layers=self._num_layers,
            dropout_rate=self._dropout_rate,
            num_residual_layers=self._num_residual_layers,
            scope=self._encoder_scope + "_PhraseLayer",
            is_training=True,  # only for dropout
            bidirectional=True,
            name=self._original_name + "_PhraseLayer",
            **self._encoder_kargs)

        modeling_layer = phrase_layer.clone(
            name=self._original_name + "_ModelingLayer")

        attention_mechanism = attentions.BiDAFAttention(
            num_units=self._num_units * 2,
            name=self._original_name + "_BidafAttention")

        # Create sequence masks
        sequence_1_mask = array_ops.sequence_mask(
            sequence_1_length, dtype=dtypes.float32,
            maxlen=array_ops.shape(sequence_1)[1])
        sequence_2_mask = array_ops.sequence_mask(
            sequence_2_length, dtype=dtypes.float32,
            maxlen=array_ops.shape(sequence_2)[1])

        # Phrase Layer
        processed_sequence_1 = phrase_layer(
            inputs=sequence_1,
            sequence_length=sequence_1_length)

        processed_sequence_2 = phrase_layer(
            inputs=sequence_2,
            sequence_length=sequence_2_length)

        # Bi-Attention Layer
        processed_sequence = attention_mechanism(
            encoded_passage=processed_sequence_1,
            encoded_question=processed_sequence_2,
            passage_mask=sequence_1_mask,
            question_mask=sequence_2_mask)

        # Modeling Layer
        processed_sequence_1 = modeling_layer(
            inputs=processed_sequence,
            sequence_length=sequence_1_length)

        return processed_sequence_1
