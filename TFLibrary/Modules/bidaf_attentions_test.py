import torch
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_util
from TFLibrary.Modules.attentions import (masked_softmax,
                                          matrix_attention,
                                          replace_masked_values)

from allennlp.nn import util
from allennlp.modules.similarity_functions.linear import LinearSimilarity
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention


BATCH_SIZE = 32
NUM_UNITS = 16
S1_LENGTH = 50
S2_LENGTH = 35
tf.enable_eager_execution()


def convert_to_numpy(X):
    if torch.is_tensor(X) and X.requires_grad:
        return X.detach().numpy()
    if torch.is_tensor(X) and not X.requires_grad:
        return X.numpy()
    if tensor_util.is_tensor(X):
        return X.numpy()
    
    return X


class AttentionModulesTest(tf.test.TestCase):

    def testPyTorchSansityCheck(self):
        passage_question_similarity = torch.randn(
            BATCH_SIZE, S1_LENGTH, S2_LENGTH)
        question_mask = torch.from_numpy(
            np.random.randint(
                0, 2, (BATCH_SIZE, S2_LENGTH)).astype(np.float32))

        # Test if these two approaches are the same
        masked_similarity_1 = util.replace_masked_values(
            passage_question_similarity, question_mask.unsqueeze(1), -1e7)
        masked_similarity_2 = replace_masked_values(
            passage_question_similarity, question_mask.unsqueeze(1), -1e7)

        self.assertAllClose(masked_similarity_1, masked_similarity_2)

    def testBiDAFAttention(self):
        # Initialization
        vec_1 = np.random.randn(BATCH_SIZE, S1_LENGTH, NUM_UNITS)
        vec_2 = np.random.randn(BATCH_SIZE, S2_LENGTH, NUM_UNITS)
        vec_1_mask = np.random.randint(
            0, 2, (BATCH_SIZE, S1_LENGTH)).astype(np.float32)
        vec_2_mask = np.random.randint(
            0, 2, (BATCH_SIZE, S2_LENGTH)).astype(np.float32)

        similarity_function, allennlp_results = AllenNLP_Methods(
            vec_1, vec_2, vec_1_mask, vec_2_mask)
        tf_results = Tensorflow_Methods(
            vec_1, vec_2, vec_1_mask, vec_2_mask,
            attention_w=convert_to_numpy(similarity_function._weight_vector),
            attention_b=convert_to_numpy(similarity_function._bias))

        self.assertEquals(len(allennlp_results), len(tf_results))
        self.assertEquals(allennlp_results.keys(), tf_results.keys())
        for key in tf_results.keys():
            self.assertAllClose(
                convert_to_numpy(tf_results[key]),
                convert_to_numpy(allennlp_results[key]),
                msg=key)


def Tensorflow_Methods(vec_1, vec_2, vec_1_mask, vec_2_mask,
                       attention_w, attention_b):
    # Initialization
    # batch_size = BATCH_SIZE
    # passage_length = S1_LENGTH
    # encoding_dim = NUM_UNITS
    encoded_passage = tf.convert_to_tensor(vec_1, dtype=tf.float32)
    encoded_question = tf.convert_to_tensor(vec_2, dtype=tf.float32)
    passage_mask = tf.convert_to_tensor(vec_1_mask, dtype=tf.float32)
    question_mask = tf.convert_to_tensor(vec_2_mask, dtype=tf.float32)

    # -----------------------------------------------------
    # Tensorflow
    # -----------------------------------------------------

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
        tf.expand_dims(question_passage_attention, axis=1), encoded_passage)
    # Shape: (batch_size, passage_length, encoding_dim)
    tiled_question_passage_vector = tf.tile(
        question_passage_vector, [1, S1_LENGTH, 1])

    # Shape: (batch_size, passage_length, encoding_dim * 4)
    final_merged_passage = tf.concat(
        [encoded_passage,
         passage_question_vectors,
         encoded_passage * passage_question_vectors,
         encoded_passage * tiled_question_passage_vector],
        axis=-1)


    return {
        "passage_question_similarity": passage_question_similarity,
        "passage_question_attention": passage_question_attention,
        "passage_question_vectors": passage_question_vectors,
        "masked_similarity": masked_similarity,
        "question_passage_similarity": question_passage_similarity,
        "question_passage_attention": question_passage_attention,
        # "question_passage_vector": question_passage_vector,
        "question_passage_vector": tf.squeeze(question_passage_vector),
        "tiled_question_passage_vector": tiled_question_passage_vector,
        "final_merged_passage": final_merged_passage}


def AllenNLP_Methods(vec_1, vec_2, vec_1_mask, vec_2_mask):
    # Initialization
    batch_size = BATCH_SIZE
    passage_length = S1_LENGTH
    encoding_dim = NUM_UNITS
    encoded_passage = torch.from_numpy(vec_1).type(torch.FloatTensor)
    encoded_question = torch.from_numpy(vec_2).type(torch.FloatTensor)
    passage_mask = torch.from_numpy(vec_1_mask).type(torch.FloatTensor)
    question_mask = torch.from_numpy(vec_2_mask).type(torch.FloatTensor)

    similarity_function = LinearSimilarity(
        NUM_UNITS, NUM_UNITS, combination="x,y,x*y")
    _matrix_attention = LegacyMatrixAttention(similarity_function)


    # -----------------------------------------------------
    # AllenNLP
    # -----------------------------------------------------

    # Shape: (batch_size, passage_length, question_length)
    passage_question_similarity = _matrix_attention(
        encoded_passage, encoded_question)
    # Shape: (batch_size, passage_length, question_length)
    passage_question_attention = util.masked_softmax(
        passage_question_similarity, question_mask)
    # Shape: (batch_size, passage_length, encoding_dim)
    passage_question_vectors = util.weighted_sum(
        encoded_question, passage_question_attention)

    # We replace masked values with something really
    # negative here, so they don't affect the
    # max below.
    masked_similarity = util.replace_masked_values(
        passage_question_similarity,
        question_mask.unsqueeze(1), -1e7)
    # Shape: (batch_size, passage_length)
    question_passage_similarity = masked_similarity.max(
        dim=-1)[0].squeeze(-1)
    # Shape: (batch_size, passage_length)
    question_passage_attention = util.masked_softmax(
        question_passage_similarity, passage_mask)
    # Shape: (batch_size, encoding_dim)
    question_passage_vector = util.weighted_sum(
        encoded_passage, question_passage_attention)
    # Shape: (batch_size, passage_length, encoding_dim)
    tiled_question_passage_vector = (
        question_passage_vector.unsqueeze(1).expand(
            batch_size, passage_length, encoding_dim))

    # Shape: (batch_size, passage_length, encoding_dim * 4)
    final_merged_passage = torch.cat(
        [encoded_passage,
         passage_question_vectors,
         encoded_passage * passage_question_vectors,
         encoded_passage * tiled_question_passage_vector],
        dim=-1)

    return similarity_function, {
        "passage_question_similarity": passage_question_similarity,
        "passage_question_attention": passage_question_attention,
        "passage_question_vectors": passage_question_vectors,
        "masked_similarity": masked_similarity,
        "question_passage_similarity": question_passage_similarity,
        "question_passage_attention": question_passage_attention,
        "question_passage_vector": question_passage_vector,
        "tiled_question_passage_vector": tiled_question_passage_vector,
        "final_merged_passage": final_merged_passage}


if __name__ == "__main__":
    tf.test.main()
