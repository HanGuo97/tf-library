import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import *
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _bahdanau_score


class CustomizedAttentionWrapper(AttentionWrapper):
    """Wraps another `RNNCell` with attention.
    """

    def __init__(self, attention_layer=None, **kargs):
        """
        Args:
           cell,
           attention_mechanism,
           attention_layer_size=None,
           alignment_history=False,
           cell_input_fn=None,
           output_attention=True,
           initial_cell_state=None,
           name=None
        Replace original attention_layers with customized one
        """
        super(CustomizedAttentionWrapper, self).__init__(**kargs)
        if attention_layer is not None:
            attention_layers = tuple(
                attention_layer
                if isinstance(attention_layer, (list, tuple))
                else (attention_layer,))

            for layer in attention_layers:
                if not isinstance(layer, layers_core.Dense):
                    raise TypeError("Expected layer to be layers_core.Dense",
                                    " found ", type(layer).__name__)

            self._attention_layers = attention_layers
            self._attention_layer_size = sum(
                [l.units for l in attention_layers])


class CustomizedBahdanauAttention(BahdanauAttention):
    """
    Args:
        num_units,
        memory,
        memory_sequence_length=None,
        normalize=False,
        probability_fn=None,
        score_mask_value=None,
        dtype=None,
        name="BahdanauAttention"
    """

    def __init__(self, scope=None, **kargs):
        super(CustomizedBahdanauAttention, self).__init__(**kargs)
        self._attention_scope = scope

    def __call__(self, query, state):
        """Score the query based on the keys and values.
        Args:
          query: Tensor of dtype matching `self.values` and shape
            `[batch_size, query_depth]`.
          state: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]`
            (`alignments_size` is memory's `max_time`).
        Returns:
          alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]` (`alignments_size` is memory's
            `max_time`).
        """
        with variable_scope.variable_scope(
                self._attention_scope, "bahdanau_attention", [query], reuse=tf.AUTO_REUSE):
            processed_query = self.query_layer(query) if self.query_layer else query
            score = _bahdanau_score(processed_query, self._keys, self._normalize)
        alignments = self._probability_fn(score, state)
        next_state = alignments
        return alignments, next_state
