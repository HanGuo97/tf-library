from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from tensorflow.python.ops import lookup_ops
from TFLibrary.Data.utils import vocab_utils
from TFLibrary.Data.utils import iterator_utils


BATCH_SIZE = 32
NUM_BATCHES_TO_TEST = 5
TEST_DATA_BASEDIR = "../test_data/multinli/test"


def build_data(data_file, batch_size, graph):
    
    src_vocab_file = data_file + ".source_vocab"
    tgt_vocab_file = data_file + ".label_vocab"

    (token_vocab_size,
     src_vocab_file) = vocab_utils.check_vocab(
        vocab_file=src_vocab_file,
        out_dir=os.path.dirname(src_vocab_file),
        check_special_token=True)

    (label_vocab_size,
     tgt_vocab_file) = vocab_utils.check_vocab(
        vocab_file=tgt_vocab_file,
        out_dir=os.path.dirname(tgt_vocab_file),
        check_special_token=False)

    tf.logging.info("token_vocab_size = %d from %s" %
        (token_vocab_size, src_vocab_file))
    tf.logging.info("label_vocab_size = %d from %s" %
        (label_vocab_size, tgt_vocab_file))

    # train dataset
    with graph.as_default():
        # vocabs are stricted to train vocabs
        src_vocab_table = lookup_ops.index_table_from_file(
            src_vocab_file, default_value=vocab_utils.UNK_ID)
        # no UNKs in target labels
        tgt_vocab_table = lookup_ops.index_table_from_file(tgt_vocab_file)

        src_1 = tf.data.TextLineDataset(data_file + ".sequence_1")
        src_2 = tf.data.TextLineDataset(data_file + ".sequence_2")
        tgt = tf.data.TextLineDataset(data_file + ".labels")
        data_batch = iterator_utils.get_pairwise_classification_iterator(
            src_dataset_1=src_1,
            src_dataset_2=src_2,
            tgt_dataset=tgt,
            src_vocab_table=src_vocab_table,
            tgt_vocab_table=tgt_vocab_table,
            batch_size=batch_size,
            sos=vocab_utils.SOS,
            eos=vocab_utils.EOS,
            random_seed=11,
            shuffle=False)


        # Additional Steps for Testing
        # ------------------------------------------
        reverse_src_vocab_table = lookup_ops.index_to_string_table_from_file(
            src_vocab_file, default_value=vocab_utils.UNK)
        reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(
            tgt_vocab_file)

        reversed_src_1 = reverse_src_vocab_table.lookup(
            tf.to_int64(data_batch.source_1))
        reversed_src_2 = reverse_src_vocab_table.lookup(
            tf.to_int64(data_batch.source_2))
        reversed_tgt = reverse_tgt_vocab_table.lookup(
            tf.to_int64(data_batch.target))
        # ------------------------------------------


    return (data_batch, token_vocab_size, label_vocab_size,
            reversed_src_1, reversed_src_2, reversed_tgt)


def filter_BatchedInput(batch):
    """sess.run(everything except batch.initializer)"""
    if not isinstance(batch, iterator_utils.BatchedInput):
        raise TypeError("`batch` must be iterator_utils.BatchedInput")
    
    batched_input_dict = OrderedDict()
    for key, val in batch._asdict().items():
        if "initializer" in key:
            continue
        batched_input_dict[key] = val

    return batched_input_dict


@np.vectorize
def remove_sos_and_eos(token):
    if token in [vocab_utils.SOS, vocab_utils.EOS]:
        return ""
    return token


def process_reversed_texts(token_array):
    """Takes tokens array, removes SOS + EOS, and formats into texts"""
    texts = []
    token_array = remove_sos_and_eos(token_array)
    for tokens in token_array.tolist():
        texts.append(" ".join(tokens).strip())
    return texts


def read_text_file(file_dir):
    with open(file_dir) as f:
        texts = [d.strip() for d in f.readlines()]
    return texts


class DataReaderTest(tf.test.TestCase):


    def testData(self):
        graph = tf.Graph()
        (data_batch,
         token_vocab_size,
         label_vocab_size,
         reversed_src_1,
         reversed_src_2,
         reversed_tgt) = build_data(
            data_file=TEST_DATA_BASEDIR,
            batch_size=BATCH_SIZE, graph=graph)

        reversed_sources_1 = []
        reversed_sources_2 = []
        reversed_targets = []

        fetched_batch_dicts = []
        data_batch_dict = filter_BatchedInput(data_batch)
        data_batch_dict["reversed_src_1"] = reversed_src_1
        data_batch_dict["reversed_src_2"] = reversed_src_2
        data_batch_dict["reversed_tgt"] = reversed_tgt
        with self.test_session(graph=graph) as session:
            session.run(tf.tables_initializer())
            session.run(tf.global_variables_initializer())
            session.run(data_batch.initializer)
            for i in range(NUM_BATCHES_TO_TEST):
                fetched = session.run(data_batch_dict)
                fetched_batch_dicts.append(fetched)

                reversed_src_1 = process_reversed_texts(
                    fetched["reversed_src_1"])
                reversed_src_2 = process_reversed_texts(
                    fetched["reversed_src_2"])
                reversed_tgt = fetched["reversed_tgt"].tolist()

                reversed_sources_1 += (reversed_src_1)
                reversed_sources_2 += (reversed_src_2)
                reversed_targets += (reversed_tgt)

        expected_sources_1 = read_text_file(TEST_DATA_BASEDIR + ".sequence_1")
        expected_sources_2 = read_text_file(TEST_DATA_BASEDIR + ".sequence_2")
        expected_targets = read_text_file(TEST_DATA_BASEDIR + ".labels")

        expected_sources_1 = expected_sources_1[:len(reversed_sources_1)]
        expected_sources_2 = expected_sources_2[:len(reversed_sources_2)]
        expected_targets = expected_targets[:len(reversed_targets)]

        self.assertEqual(expected_sources_1, reversed_sources_1, "sources_1")
        self.assertEqual(expected_sources_2, reversed_sources_2, "sources_2")
        self.assertEqual(expected_targets, reversed_targets, "targets")

        tgt_lens = np.stack(
            d["target_sequence_length"]
            for d in fetched_batch_dicts)
        expected_tgt_lens = np.ones_like(tgt_lens)
        self.assertAllEqual(expected_tgt_lens, tgt_lens, "targets")


if __name__ == "__main__":
    tf.test.main()
