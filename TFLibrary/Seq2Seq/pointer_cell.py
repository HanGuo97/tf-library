from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import core as layers_core

from tensorflow.python.layers import core as core_layers
from tensorflow.contrib.seq2seq import AttentionMechanism

import numpy as np
from collections import namedtuple
from TFLibrary.Seq2Seq import attention_utils
from TFLibrary.Seq2Seq import rnn_cell_utils

_zero_state_tensors = rnn_cell_impl._zero_state_tensors


def masked_attention(score, enc_padding_mask):
    raise NotImplementedError("Not Tested")
    """Softmax + enc_padding_mask + re-normalize"""
    # take softmax. shape (batch_size, attn_length)
    attn_dist = nn_ops.softmax(score)
    attn_dist *= enc_padding_mask
    # shape (batch_size)
    masked_sums = math_ops.reduce_sum(attn_dist, axis=1)
    # re-normalize
    return attn_dist / array_ops.reshape(masked_sums, [-1, 1])


def _calc_final_dist(vocab_dist, attn_dist, p_gen,
                     batch_size, vocab_size, num_source_OOVs,
                     enc_batch_extended_vocab):
    # P(gen) x P(vocab)
    weighted_P_vocab = p_gen * vocab_dist
    # (1 - P(gen)) x P(attention)
    weighted_P_copy = (1 - p_gen) * attn_dist

    # get the word-idx for all words
    extended_vsize = vocab_size + num_source_OOVs
    # placeholders to OOV words
    extra_zeros = array_ops.zeros((batch_size, num_source_OOVs))
    # this distribution span the entire words
    weighted_P_vocab_extended = array_ops.concat(
        axis=1, values=[weighted_P_vocab, extra_zeros])

    # assign probabilities from copy distribution
    # into correspodning positions in extended_vocab_dist

    # to do this, we need to use scatter_nd
    # scatter_nd (in this case) requires two numbers
    # one is the index in batch-dimension
    # the other is the index in vocab-dimension
    # So first, we create a batch-matrix like:
    # [[1, 1, 1, 1, 1, ...],
    #  [2, 2, 2, 2, 2, ...],
    #  [...]
    #  [N, N, N, N, N, ...]]

    # [1, 2, ..., N]
    # to [[1], [2], ..., [N]]
    # and finally to the final shape
    enc_seq_len = array_ops.shape(enc_batch_extended_vocab)[1]
    batch_nums = math_ops.range(0, limit=batch_size)
    batch_nums = array_ops.expand_dims(batch_nums, 1)
    batch_nums = array_ops.tile(batch_nums, [1, enc_seq_len])

    # stick together batch-dim and index-dim
    indices = array_ops.stack((batch_nums, enc_batch_extended_vocab), axis=2)
    scatter_shape = [batch_size, extended_vsize]
    # scatter the attention distributions
    # into the word-indices
    weighted_P_copy_projected = array_ops.scatter_nd(
        indices, weighted_P_copy, scatter_shape)

    # Add the vocab distributions and the copy distributions together
    # to get the final distributions, final_dists is a list length
    # max_dec_steps; each entry is (batch_size, extended_vsize)
    # giving the final distribution for that decoder timestep
    # Note that for decoder timesteps and examples corresponding to
    # a [PAD] token, this is junk - ignore.
    final_dists = weighted_P_vocab_extended + weighted_P_copy_projected

    return final_dists


PointerWrapperState = namedtuple(
    "PointerWrapperState",
    ("cell_state", "time",  # RNN Cell
    "attention", "alignments", "alignment_history",  # Attention
    "p_gen", "coverage", "p_gen_history", "coverage_history",
    "logits_history", "vocab_dists_history", "final_dists_history"))  # Pointer


class PointerWrapper(rnn_cell_impl.RNNCell):

    def __init__(self,
                 cell,
                 # attention
                 attention_mechanism,
                 attention_layer_size=None,
                 alignment_history=False,
                 cell_input_fn=None,
                 output_attention=True,
                 # pointer
                 coverage=False,
                 batch_size=None,
                 vocab_size=None,
                 num_source_OOVs=None,
                 enc_batch_extended_vocab=None,
                 probability_fn=None,
                 # misc
                 initial_cell_state=None,
                 name=None,
                 attention_scope=None,
                 pointer_scope=None):

        super(PointerWrapper, self).__init__(name=name)
        

        # some todo's
        # if initial_state_attention:
        #     raise NotImplementedError
        if coverage:
            print("there is a problem in coverage,"
                  "see Line 216 - 221 in original"
                  "attention_decoder")

        if initial_cell_state:
            raise NotImplementedError("Not Supported")

        if not rnn_cell_impl._like_rnncell(cell):
            raise TypeError(
                "cell must be an RNNCell, saw type: %s" % type(cell).__name__)

        if not isinstance(attention_mechanism, AttentionMechanism):
            raise TypeError(
                "attention_mechanism must be an AttentionMechanism or list of "
                "multiple AttentionMechanism instances, saw type: %s"
                % type(attention_mechanism).__name__)

        
        if cell_input_fn is None:
            cell_input_fn = (lambda inputs, attention:
                             array_ops.concat([inputs, attention], -1))
        if probability_fn is None:
            probability_fn = nn_ops.softmax

        if not callable(cell_input_fn):
            raise TypeError("cell_input_fn must be callable")
        if not callable(probability_fn):
            raise TypeError("probability_fn should be callable")

        # assert this for now, but can be relaxed later
        if attention_layer_size != cell.output_size:
            raise ValueError("attention_layer_size != cell.output_size")


        self._build_layers(
            num_units=attention_layer_size,
            vocab_size=vocab_size,
            dtype=attention_mechanism.dtype)

        self._cell = cell
        self._attention_mechanism = attention_mechanism
        self._attention_layer_size = attention_layer_size
        self._cell_input_fn = cell_input_fn
        self._output_attention = output_attention
        self._alignment_history = alignment_history

        self._coverage = coverage
        self._pointer_scope = pointer_scope
        self._attention_scope = attention_scope

        self._vocab_size = vocab_size
        self._batch_size = batch_size
        self._probability_fn = probability_fn
        self._num_source_OOVs = num_source_OOVs
        self._enc_batch_extended_vocab = enc_batch_extended_vocab

    def _build_layers(self, num_units, vocab_size, dtype):
        # layers
        
        # attention_layer maps cell_outputs into attention outputs
        attention_layer = layers_core.Dense(
            units=num_units, use_bias=False,
            dtype=dtype, name="attention_layer")

        # pgen_layer maps states into p_gen
        pgen_layer = core_layers.Dense(
            units=1, activation=math_ops.sigmoid,
            use_bias=True, dtype=dtype, name="pgen_layer")

        # coverage kernels transforms coverage vector
        coverage_kernel = core_layers.Dense(
            units=num_units, use_bias=False,
            dtype=dtype, name="coverage_kernel")

        logits_layer = core_layers.Dense(
            units=vocab_size, use_bias=True,
            dtype=dtype, name="logits_layer")

        self._pgen_layer = pgen_layer
        self._coverage_kernel = coverage_kernel
        self._logits_layer = logits_layer
        self._attention_layer = attention_layer



    @property
    def output_size(self):
        if self._output_attention:
            return self._attention_layer_size
        else:
            return self._cell.output_size

    @property
    def state_size(self):
        """The `state_size` property of `PointerWrapperState`.
        Returns:
            An `PointerWrapperState` of shapes used by this object.
        """
        return PointerWrapperState(
            # Cells
            cell_state=self._cell.state_size,
            time=tensor_shape.TensorShape([]),
            # Attentions
            attention=self._attention_layer_size,
            alignments=self._attention_mechanism.alignments_size,
            alignment_history=(),  # sometimes a TensorArray
            # Pointers
            # p_gen \in [0,1]
            p_gen=tensor_shape.TensorShape([1]),
            # coverage = sum(alignments)
            coverage=self._attention_mechanism.alignments_size,
            p_gen_history=(),
            coverage_history=(),
            logits_history=(),
            vocab_dists_history=(),
            final_dists_history=())

    def zero_state(self, batch_size, dtype):
        """Return an initial (zero) state tuple for this `AttentionWrapper`.
        **NOTE** Please see the initializer documentation for details of how
        to call `zero_state` if using an `AttentionWrapper` with a
        `BeamSearchDecoder`.
        Args:
          batch_size: `0D` integer tensor: the batch size.
          dtype: The internal state data type.
        Returns:
          An `AttentionWrapperState` tuple containing zeroed out tensors and,
          possibly, empty `TensorArray` objects.
        Raises:
          ValueError: (or, possibly at runtime, InvalidArgument), if
            `batch_size` does not match the output size of the encoder passed
            to the wrapper object at initialization time.
        """
        zs_name = type(self).__name__ + "ZeroState"
        with ops.name_scope(zs_name, values=[batch_size]):
            zero_cell_state = self._cell.zero_state(batch_size, dtype)
            zero_attention = _zero_state_tensors(
                state_size=self._attention_layer_size,
                batch_size=batch_size,
                dtype=dtype)
            zero_alignments = self._attention_mechanism.initial_alignments(
                batch_size=batch_size, dtype=dtype)
            zero_coverage = self._attention_mechanism.initial_alignments(
                batch_size=batch_size, dtype=dtype)

            # all kinds of histories
            zero_alignment_history = (tensor_array_ops.TensorArray(
                dtype=dtype, size=0, dynamic_size=True)
                if self._alignment_history else ())
            zero_p_gen_history = (tensor_array_ops.TensorArray(
                dtype=dtype, size=0, dynamic_size=True)
                if self._alignment_history else ())
            zero_coverage_history = (tensor_array_ops.TensorArray(
                dtype=dtype, size=0, dynamic_size=True)
                if self._alignment_history else ())
            zero_logits_history = (tensor_array_ops.TensorArray(
                dtype=dtype, size=0, dynamic_size=True)
                if self._alignment_history else ())
            zero_vocab_dists_history = (tensor_array_ops.TensorArray(
                dtype=dtype, size=0, dynamic_size=True)
                if self._alignment_history else ())
            zero_final_dists_history = (tensor_array_ops.TensorArray(
                dtype=dtype, size=0, dynamic_size=True)
                if self._alignment_history else ())

            return PointerWrapperState(
                # Cells
                cell_state=zero_cell_state,
                time=array_ops.zeros([], dtype=dtypes.int32),
                # Attention
                attention=zero_attention,
                alignments=zero_alignments,
                alignment_history=zero_alignment_history,
                # Pointers
                p_gen=array_ops.zeros([batch_size, 1], dtype=dtypes.float32),
                coverage=zero_coverage,
                p_gen_history=zero_p_gen_history,
                coverage_history=zero_coverage_history,
                logits_history=zero_logits_history,
                vocab_dists_history=zero_vocab_dists_history,
                final_dists_history=zero_final_dists_history)

    def call(self, inputs, state):
        if not isinstance(state, PointerWrapperState):
            raise TypeError("Expected state to be instance "
                            "of AttentionWrapperState. "
                            "Received type %s instead." % type(state))

        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)
        last_layer_state = rnn_cell_utils.get_last_layer_cell_state(
            cell_states=next_cell_state)

        # Step 2: Compute attention and write to history
        with variable_scope.variable_scope(self._attention_scope, "Attention"):

            # attention_utils._compute_attention calls the function
            # attention_mechanism.__call__(self, query, state)
            # where state is not used.
            # In the normal attention setting, we have
            #   e = v^T tanh (W_h x h + W_s x s + b)
            # where h is the query and s is the memory
            # In the coverage-attention setting, we have
            #   e = v^T tanh (W_h x h + W_s x s + + W_c x c b)
            # where c is the coverage
            # A simple reparameterization can be
            #   e = v^T tanh (W_h' x h' + W_s x s + b)
            # where h' = [h;c], and this makes implementation cleaner
            if self._coverage:
                attention_query = array_ops.concat(
                    [cell_output, state.coverage], axis=-1)
            else:
                # copy coverage to next time step
                coverage = state.coverage
                attention_query = cell_output

            # calculate attentions
            (attention, alignments, _, context) = (
                attention_utils._compute_attention(
                    attention_mechanism=self._attention_mechanism,
                    cell_output=attention_query,
                    attention_state=None,
                    attention_layer=(self._attention_layer
                        if self._attention_layer else None)))

            if self._coverage:
                # coverage = sum of alignments
                # up until previous time step
                coverage = state.coverage + alignments


        # Step 3: Compute pointer and coverage and write to histories
        with variable_scope.variable_scope(self._pointer_scope, "Pointer"):

            p_gen = self._pgen_layer(array_ops.concat([
                context, last_layer_state.c,
                last_layer_state.h, cell_inputs], -1))


        # Step 4: calculating final distributions
        logits = self._logits_layer(attention)
        vocab_dist = self._probability_fn(logits)
        
        final_dist = _calc_final_dist(
            vocab_dist=vocab_dist,
            attn_dist=alignments,
            p_gen=p_gen,
            batch_size=self._batch_size,
            vocab_size=self._vocab_size,
            num_source_OOVs=self._num_source_OOVs,
            enc_batch_extended_vocab=self._enc_batch_extended_vocab)


        # Step 5: log histories
        time = state.time
        alignment_history = state.alignment_history.write(
            time, alignments) if self._alignment_history else ()
        coverage_history = state.coverage_history.write(time, coverage)
        p_gen_history = state.p_gen_history.write(time, p_gen)
        logits_history = state.logits_history.write(time, logits)
        vocab_dists_history = state.vocab_dists_history.write(time, vocab_dist)
        final_dists_history = state.final_dists_history.write(time, final_dist)

        
        next_state = PointerWrapperState(
            # Cells
            time=state.time + 1,
            cell_state=next_cell_state,
            # Attention
            attention=attention,
            alignments=alignments,
            alignment_history=alignment_history,
            # Pointer
            p_gen=p_gen,
            coverage=coverage,
            p_gen_history=p_gen_history,
            coverage_history=coverage_history,
            logits_history=logits_history,
            vocab_dists_history=vocab_dists_history,
            final_dists_history=final_dists_history)

        if self._output_attention:
            return attention, next_state
        else:
            return cell_output, next_state


def unstack_and_transpose_histories(final_loop_state):
    # [time, batch, nun_units] to [batch, time, num_units]
    alignments = array_ops.transpose(
        final_loop_state.alignment_history.stack(), perm=[1, 0, 2])
    p_gens = array_ops.transpose(
        final_loop_state.p_gen_history.stack(), perm=[1, 0, 2])
    logits = array_ops.transpose(
        final_loop_state.logits_history.stack(), perm=[1, 0, 2])
    vocab_dists = array_ops.transpose(
        final_loop_state.vocab_dists_history.stack(), perm=[1, 0, 2])
    final_dists = array_ops.transpose(
        final_loop_state.final_dists_history.stack(), perm=[1, 0, 2])

    return (alignments, p_gens, logits, vocab_dists, final_dists)


def _unnecessary_test():
    Wa = np.random.random([128, 128])
    Wb = np.random.random([128, 128])
    A = np.random.random([128])
    B = np.random.random([128])

    W = np.concatenate([Wa, Wb], axis=-1)
    X = np.concatenate([A, B], axis=-1)

    passed = (np.matmul(Wa, A) + np.matmul(Wb, B) - np.matmul(W, X)) < 1e-11
    print("Passed: ", passed.all())
