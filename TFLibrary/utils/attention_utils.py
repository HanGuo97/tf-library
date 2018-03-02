from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from tensorflow.python.layers import base as layers_base
from tensorflow.python.layers import core as layers_core
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.contrib.framework.python.framework import tensor_util
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _bahdanau_score
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _BaseAttentionMechanism


def masked_softmax(score, enc_padding_mask):
    attn_dist = nn_ops.softmax(score)
    attn_dist *= enc_padding_mask
    masked_sums = math_ops.reduce_sum(attn_dist, axis=1)
    # re-normalize
    return attn_dist / array_ops.reshape(masked_sums, [-1, 1])


class BahdanauAttentionTester(_BaseAttentionMechanism):
    """General Attention Tester

    Built upon Tensorflow's BahdanauAttention
    but allow customized layer and masked attention
    """
    def __init__(self,
                 num_units,
                 memory,
                 mask,
                 normalize=False,
                 # for test
                 query_layer=None,
                 memory_layer=None,
                 probability_fn=None,
                 # others
                 scope=None,
                 name="BahdanauAttentionTest",):
        
        if not probability_fn:
            probability_fn = lambda score, _: masked_softmax(score, mask)

        if not callable(probability_fn):
            raise TypeError("probability_fn must be callable")
        
        super(BahdanauAttentionTester, self).__init__(
            query_layer=query_layer,
            memory_layer=memory_layer,
            memory=memory,
            probability_fn=probability_fn,
            memory_sequence_length=None,
            score_mask_value=None,
            name=name)

        self._num_units = num_units
        self._normalize = normalize
        self._name = name
        self._attention_scope = scope
    
    def __call__(self, query, state=None):
        with vs.variable_scope(self._attention_scope, reuse=True):
            processed_query = (self.query_layer(query)
                              if self.query_layer else query)
            score = _bahdanau_score(
                processed_query=processed_query,
                keys=self._keys, normalize=self._normalize)
        alignments = self._probability_fn(score, state)
        next_state = alignments
        return alignments, next_state
