import tensorflow as tf
from TFLibrary.Modules import base
from TFLibrary.Modules import utils


def masked_softmax(X, mask, epsilon=1e-13):
    if mask is None:
        return tf.nn.softmax(X)
    else:
        # To limit numerical errors from large vector elements
        # outside mask, we zero these out
        result = tf.nn.softmax(X * mask)
        result = result * mask
        result = result / (tf.reduce_sum(
            result, axis=-1, keep_dims=True) + epsilon)
        return result


def matrix_attention(X1, X2, w, b):
    """w^T [X1, X2, X1 * X2] + b"""
    # [batch_size, seq1_length, seq2_length, num_units]
    seq1_length = tf.shape(X1)[1]
    seq2_length = tf.shape(X2)[1]
    _X1 = tf.expand_dims(X1, axis=2)
    _X1 = tf.tile(_X1, [1, 1, seq2_length, 1])
    _X2 = tf.expand_dims(X2, axis=1)
    _X2 = tf.tile(_X2, [1, seq1_length, 1, 1])
    # [X1, X2, X1 * X2]
    combined_vectors = tf.concat([_X1, _X2, _X1 * _X2], axis=-1)
    # w^T X + b
    M = tf.reduce_sum(combined_vectors * w, axis=-1)
    # `M = tf.nn.bias_add(M, b)` cannot broadcast last dimension
    M = M + b
    return M


def replace_masked_values(X, mask, value):
    # similar to torch.tensor.masked_fill
    return X * mask + (1 - mask) * value


class CrossAttention(base.AbstractModule):
    """Tensorflow Version of
    https://github.com/jsalt18-sentence-repl/jiant/blob/master/src/utils.py"""

    def __init__(self, name="cross_attention"):
        super(CrossAttention, self).__init__(name=name)

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


class BiDAFAttention(base.AbstractModule):
    """Tensorflow Version of
    https://github.com/jsalt18-sentence-repl/jiant/blob/master/src/utils.py"""

    def __init__(self, num_units, name="bidaf_attention"):
        super(BiDAFAttention, self).__init__(name=name)
        self._num_units = num_units

    def _build(self,
               encoded_passage,
               encoded_question,
               passage_mask,
               question_mask):

        # used in later steps
        passage_length = tf.shape(encoded_passage)[1]

        # create variable
        attention_w = tf.get_variable(
            name="attention_w",
            shape=[self._num_units],
            initializer=utils.create_linear_initializer(self._num_units))

        attention_b = tf.get_variable(
            name="attention_b",
            shape=[1],
            initializer=utils.create_bias_initializer(self._num_units))

        # Shape: (batch_size, passage_length, question_length)
        passage_question_similarity = matrix_attention(
            encoded_passage, encoded_question,
            attention_w, attention_b)
        # Shape: (batch_size, passage_length, question_length)
        passage_question_attention = masked_softmax(
            passage_question_similarity,
            tf.expand_dims(question_mask, axis=1))
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_question_vectors = tf.matmul(
            passage_question_attention, encoded_question)

        # We replace masked values with something really
        # negative here, so they don't affect the
        # max below.
        masked_similarity = replace_masked_values(
            passage_question_similarity,
            mask=tf.expand_dims(question_mask, axis=1), value=-1e7)

        # Shape: (batch_size, passage_length)
        question_passage_similarity = tf.reduce_max(
            masked_similarity, axis=-1)
        # Shape: (batch_size, passage_length)
        question_passage_attention = masked_softmax(
            question_passage_similarity, passage_mask)
        # Shape: (batch_size, 1, encoding_dim)
        question_passage_vector = tf.matmul(
            tf.expand_dims(question_passage_attention, axis=1),
            encoded_passage)
        # Shape: (batch_size, passage_length, encoding_dim)
        tiled_question_passage_vector = tf.tile(
            question_passage_vector, [1, passage_length, 1])

        # Shape: (batch_size, passage_length, encoding_dim * 4)
        final_merged_passage = tf.concat(
            [encoded_passage,
             passage_question_vectors,
             encoded_passage * passage_question_vectors,
             encoded_passage * tiled_question_passage_vector],
            axis=-1)

        return final_merged_passage

    def _clone(self, name):
        return type(self)(num_units=self._num_units,
                          name=name)
