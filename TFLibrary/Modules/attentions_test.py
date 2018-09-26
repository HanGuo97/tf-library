import torch
import numpy as np
import tensorflow as tf
from allennlp.nn import util as allennlp_utils
from allennlp.modules.matrix_attention import DotProductMatrixAttention

BATCH_SIZE = 32
NUM_UNITS = 16
S1_LENGTH = 50
S2_LENGTH = 35
tf.enable_eager_execution()


class AttentionModulesTest(tf.test.TestCase):
    def testTorchBMMAndTFMatMul(self):
        """Test Whether Torch.bmm() == tf.matmul under different precisions"""
        random_array_1 = np.random.randn(32, 64, 128)
        random_array_2 = np.random.randn(32, 128, 512)

        # torch
        torch_matmul_32 = torch.bmm(
            torch.from_numpy(random_array_1).type(torch.FloatTensor),
            torch.from_numpy(random_array_2).type(torch.FloatTensor)).numpy()

        torch_matmul_64 = torch.bmm(
            torch.from_numpy(random_array_1).type(torch.DoubleTensor),
            torch.from_numpy(random_array_2).type(torch.DoubleTensor)).numpy()

        # TF
        tf_matmul_32 = tf.matmul(
            tf.convert_to_tensor(random_array_1, dtype=tf.float32),
            tf.convert_to_tensor(random_array_2, dtype=tf.float32)).numpy()

        tf_matmul_64 = tf.matmul(
            tf.convert_to_tensor(random_array_1, dtype=tf.float64),
            tf.convert_to_tensor(random_array_2, dtype=tf.float64)).numpy()


        self.assertNotAllClose(tf_matmul_32, tf_matmul_64)
        self.assertNotAllClose(torch_matmul_32, torch_matmul_64)
        self.assertNotAllClose(tf_matmul_32, torch_matmul_32)
        self.assertAllClose(tf_matmul_64, torch_matmul_64)

    def testSimilarityMatrix(self):
        sequence_1 = np.random.randn(BATCH_SIZE, S1_LENGTH, NUM_UNITS)
        sequence_2 = np.random.randn(BATCH_SIZE, S2_LENGTH, NUM_UNITS)
        
        # AllenNLP
        similarity_mat_JIANT = DotProductMatrixAttention()(
            to_Torch_DoubleTensor(sequence_2),
            to_Torch_DoubleTensor(sequence_1)).numpy()

        # TF
        similarity_mat_TF = tf.matmul(
            sequence_2, sequence_1,
            transpose_b=True).numpy()

        self.assertAllClose(similarity_mat_TF, similarity_mat_JIANT)

    def testLastDimSoftmax(self):
        # matrix = np.random.randn(BATCH_SIZE, S2_LENGTH, S1_LENGTH)
        sequence_1 = np.random.randn(BATCH_SIZE, S1_LENGTH, NUM_UNITS)
        sequence_2 = np.random.randn(BATCH_SIZE, S2_LENGTH, NUM_UNITS)
        sequence_1_mask = np.random.randint(
            0, 2, (BATCH_SIZE, S1_LENGTH)).astype(np.float64)
        sequence_2_mask = np.random.randint(
            0, 2, (BATCH_SIZE, S2_LENGTH)).astype(np.float64)
        similarity_mat = tf.matmul(
            sequence_2, sequence_1,
            transpose_b=True).numpy()

        # JIANT
        s2_s1_attn_JIANT = JIANT_last_dim_softmax(
            to_Torch_FloatTensor(similarity_mat),
            to_Torch_FloatTensor(sequence_1_mask)).numpy()
        s1_s2_attn_JIANT = JIANT_last_dim_softmax(
            to_Torch_FloatTensor(similarity_mat).transpose(1, 2).contiguous(),
            to_Torch_FloatTensor(sequence_2_mask)).numpy()

        # AllenNLP's last_dim_softmax == masked_softmax
        s2_s1_attn_AllenNLP = allennlp_utils.masked_softmax(
            to_Torch_FloatTensor(similarity_mat),
            to_Torch_FloatTensor(sequence_1_mask)).numpy()

        s1_s2_attn_AllenNLP = allennlp_utils.masked_softmax(
            to_Torch_FloatTensor(similarity_mat).transpose(1, 2).contiguous(),
            to_Torch_FloatTensor(sequence_2_mask)).numpy()

        # TF
        s2_s1_attn_TF = TF_masked_softmax(
            similarity_mat,
            tf.expand_dims(sequence_1_mask, axis=1)).numpy()
        s1_s2_attn_TF = TF_masked_softmax(
            tf.transpose(similarity_mat, perm=(0, 2, 1)),
            tf.expand_dims(sequence_2_mask, axis=1)).numpy()


        self.assertAllClose(s2_s1_attn_TF, s2_s1_attn_AllenNLP)
        self.assertAllClose(s2_s1_attn_TF, s2_s1_attn_JIANT)
        self.assertAllClose(s1_s2_attn_TF, s1_s2_attn_AllenNLP)
        self.assertAllClose(s1_s2_attn_TF, s1_s2_attn_JIANT)

    def testWeightedSum(self):
        sequence_1 = np.random.randn(BATCH_SIZE, S1_LENGTH, NUM_UNITS)
        sequence_2 = np.random.randn(BATCH_SIZE, S2_LENGTH, NUM_UNITS)
        sequence_1_mask = np.random.randint(
            0, 2, (BATCH_SIZE, S1_LENGTH)).astype(np.float64)
        sequence_2_mask = np.random.randint(
            0, 2, (BATCH_SIZE, S2_LENGTH)).astype(np.float64)
        similarity_mat = tf.matmul(
            sequence_2, sequence_1,
            transpose_b=True).numpy()
        s2_s1_attn = TF_masked_softmax(
            similarity_mat,
            tf.expand_dims(sequence_1_mask, axis=1)).numpy()
        s1_s2_attn = TF_masked_softmax(
            tf.transpose(similarity_mat, perm=(0, 2, 1)),
            tf.expand_dims(sequence_2_mask, axis=1)).numpy()

        # JIANT
        s2_s1_vectors_JIANT = JIANT_weighted_sum(
            to_Torch_DoubleTensor(sequence_1),
            to_Torch_DoubleTensor(s2_s1_attn)).numpy()
        s1_s2_vectors_JIANT = JIANT_weighted_sum(
            to_Torch_DoubleTensor(sequence_2),
            to_Torch_DoubleTensor(s1_s2_attn)).numpy()

        # AllenNLP
        s2_s1_vectors_AllenNLP = allennlp_utils.weighted_sum(
            to_Torch_DoubleTensor(sequence_1),
            to_Torch_DoubleTensor(s2_s1_attn)).numpy()
        s1_s2_vectors_AllenNLP = allennlp_utils.weighted_sum(
            to_Torch_DoubleTensor(sequence_2),
            to_Torch_DoubleTensor(s1_s2_attn)).numpy()

        # TF
        s2_s1_vectors_TF = tf.matmul(s2_s1_attn, sequence_1).numpy()
        s1_s2_vectors_TF = tf.matmul(s1_s2_attn, sequence_2).numpy()


        self.assertAllClose(s2_s1_vectors_TF, s2_s1_vectors_AllenNLP)
        self.assertAllClose(s2_s1_vectors_TF, s2_s1_vectors_JIANT)
        self.assertAllClose(s1_s2_vectors_TF, s1_s2_vectors_AllenNLP)
        self.assertAllClose(s1_s2_vectors_TF, s1_s2_vectors_JIANT)


def to_Torch_DoubleTensor(X):
    return torch.from_numpy(X).type(torch.DoubleTensor)


def to_Torch_FloatTensor(X):
    return torch.from_numpy(X).type(torch.FloatTensor)


def TF_masked_softmax(X, mask, epsilon=1e-13):
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

# ==========================================
# JIANT's PyTorch Implementations
# ==========================================


def JIANT_last_dim_softmax(tensor, mask=None):
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor.size()[-1])
    if mask is not None:
        while mask.dim() < tensor.dim():
            mask = mask.unsqueeze(1)
        mask = mask.expand_as(tensor).contiguous()  # .float()
        mask = mask.view(-1, mask.size()[-1])
    reshaped_result = JIANT_masked_softmax(reshaped_tensor, mask)
    return reshaped_result.view(*tensor_shape)


def JIANT_masked_softmax(vector, mask):
    if mask is None:
        result = torch.nn.functional.softmax(vector)
    else:
        # To limit numerical errors from large vector
        # elements outside mask, we zero these out
        result = torch.nn.functional.softmax(vector * mask)
        result = result * mask
        result = result / (result.sum(dim=1, keepdim=True) + 1e-13)
    return result


def JIANT_weighted_sum(matrix, attention):
    # We'll special-case a few settings here,
    # where there are efficient (but poorly-named)
    # operations in pytorch that already do the computation we need.
    if attention.dim() == 2 and matrix.dim() == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    if attention.dim() == 3 and matrix.dim() == 3:
        return attention.bmm(matrix)
    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for i in range(attention.dim() - matrix.dim() + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)


if __name__ == "__main__":
    tf.test.main()
