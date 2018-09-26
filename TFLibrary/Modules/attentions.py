from warnings import warn
import tensorflow as tf
from TFLibrary.Modules import base


def masked_softmax(X, mask, epsilon=1e-13):
    if mask is None:
        return tf.nn.softmax(X)
    else:
        # To limit numerical errors from large vector elements
        # outside mask, we zero these out
        result = tf.nn.softmax(X * mask)
        result = result * mask
        result = result / (tf.reduce_sum(
            result, axis=1, keep_dims=True) + epsilon)
        return result


class CrossAttention(base.AbstractModule):
    """Tensorflow Version of
    https://github.com/jsalt18-sentence-repl/jiant/blob/master/src/utils.py"""

    def __init__(self, name="cross_attention"):
        super(CrossAttention, self).__init__(name=name)
        warn("Not Unit Test Has Been Done To Ensure Correctness")

    def _build(self, sequence_1, sequence_2, sequence_1_mask, sequence_2_mask):
        # Similarity matrix
        # Shape: (batch_size, s2_length, s1_length)
        similarity_mat = tf.matmul(sequence_2, sequence_1, transpose_b=True)

        # s2 representation
        # Shape: (batch_size, s2_length, s1_length)
        seq2_seq1_attn = masked_softmax(
            similarity_mat,
            # (batch_size, s1_length) --> (batch_size, 1, s1_length)
            tf.expand_dims(sequence_1_mask, axis=1))
        # Shape: (batch_size, s2_length, encoding_dim)
        seq2_seq1_vectors = tf.matmul(seq2_seq1_attn, sequence_1)
        # batch_size, seq_len, 4*enc_dim
        seq2_w_context = tf.concat([sequence_2, seq2_seq1_vectors], axis=2)

        # s1 representation, using same attn method
        seq1_seq2_attn = masked_softmax(
            tf.transpose(similarity_mat, perm=(0, 2, 1)),
            # (batch_size, s2_length) --> (batch_size, 1, s2_length)
            tf.expand_dims(sequence_2_mask, axis=1))
        # Shape: (batch_size, s1_length, encoding_dim)
        seq1_seq2_vectors = tf.matmul(seq1_seq2_attn, sequence_2)
        seq1_w_context = tf.concat([sequence_1, seq1_seq2_vectors], axis=2)

        return seq1_w_context, seq2_w_context

    def _clone(self, name):
        return type(self)(name=name)
