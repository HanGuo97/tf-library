"""
https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py
"""
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops


def _compute_attention(attention_mechanism,
                       cell_output,
                       attention_state,
                       attention_layer):
    """Computes the attention and alignments
       for a given attention_mechanism.
    """
    alignments, next_attention_state = attention_mechanism(
        cell_output, state=attention_state)

    # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
    expanded_alignments = array_ops.expand_dims(alignments, 1)
    # Context is the inner product of alignments and values along the
    # memory time dimension.
    # alignments shape is
    #   [batch_size, 1, memory_time]
    # attention_mechanism.values shape is
    #   [batch_size, memory_time, memory_size]
    # the batched matmul is over memory_time, so the output shape is
    #   [batch_size, 1, memory_size].
    # we then squeeze out the singleton dim.
    context = math_ops.matmul(expanded_alignments, attention_mechanism.values)
    context = array_ops.squeeze(context, [1])

    if attention_layer is not None:
        attention = attention_layer(
            array_ops.concat([cell_output, context], 1))
    else:
        attention = context

    return attention, alignments, next_attention_state, context
