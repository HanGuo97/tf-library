from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from TFLibrary.Modules import (transformer,
                               transformer_utils,
                               transformer_utils_openai)

from tensor2tensor.layers import common_attention as T2T

BATCH_SIZE = 32
SEQUENCE_LENGTH = 10
NUM_UNITS = 128
NUM_HEADS = 128 // 64


class TransformerTest(tf.test.TestCase):

    def testMultiheadAttention(self):
        graph = tf.Graph()
        with graph.as_default():
            inputs = tf.random_normal([BATCH_SIZE, SEQUENCE_LENGTH, NUM_UNITS])
            OpenAI_outputs = transformer_utils_openai.attn(
                inputs,
                scope="OpenAI",
                n_state=NUM_UNITS,
                n_head=NUM_HEADS,
                train=True, scale=True)

            T2T_outputs = T2T.multihead_attention(
                inputs,
                memory_antecedent=None,
                bias=None,
                total_key_depth=NUM_UNITS,
                total_value_depth=NUM_UNITS,
                output_depth=NUM_UNITS,
                num_heads=NUM_HEADS,
                dropout_rate=0.0,
                name="T2T")

            outputs = transformer_utils.multihead_attention(
                inputs,
                memory_antecedent=None,
                total_key_depth=NUM_UNITS,
                total_value_depth=NUM_UNITS,
                output_depth=NUM_UNITS,
                num_heads=NUM_HEADS,
                dropout_rate=0.0,
                name="Transformer")

            # pre-set matrics
            Attn_Q = np.random.random([128, 128])
            Attn_K = np.random.random([128, 128])
            Attn_V = np.random.random([128, 128])
            Proj = np.random.random([128, 128])

        with self.test_session(graph=graph) as sess:
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            results_openai = sess.run(
                tf.equal(outputs, OpenAI_outputs), feed_dict={
                    # OpenAI's Implementation
                    "OpenAI/c_attn/w:0": np.expand_dims(np.concatenate(
                        # (1, num_units, num_units x 3), three matrics stacked
                        [Attn_Q, Attn_K, Attn_V], axis=-1), axis=0),
                    # setting biases to all zeros
                    "OpenAI/c_attn/b:0": np.zeros([384, ]),
                    "OpenAI/c_proj/w:0": np.expand_dims(Proj, axis=0),
                    "OpenAI/c_proj/b:0": np.zeros([128, ]),

                    # our own implementation
                    "Transformer/q/kernel:0": Attn_Q,
                    "Transformer/k/kernel:0": Attn_K,
                    "Transformer/v/kernel:0": Attn_V,
                    "Transformer/output_transform/kernel:0": Proj})

            results_t2t = sess.run(
                tf.equal(outputs, T2T_outputs), feed_dict={
                    # Tensor2Tensor's Implementation
                    "T2T/q/kernel:0": Attn_Q,
                    "T2T/k/kernel:0": Attn_K,
                    "T2T/v/kernel:0": Attn_V,
                    "T2T/output_transform/kernel:0": Proj,

                    # our own implementation
                    "Transformer/q/kernel:0": Attn_Q,
                    "Transformer/k/kernel:0": Attn_K,
                    "Transformer/v/kernel:0": Attn_V,
                    "Transformer/output_transform/kernel:0": Proj})

            self.assertTrue(results_openai.all())
            self.assertTrue(results_t2t.all())


if __name__ == "__main__":
    tf.test.main()
