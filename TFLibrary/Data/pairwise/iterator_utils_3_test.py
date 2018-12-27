from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import lookup_ops
from TFLibrary.utils import misc_utils
from TFLibrary.Data.pairwise import iterator_utils_3
from constants import (CACHED_ELMO_NUM_ELEMENTS,
                       CACHED_ELMO_NUM_UNITS,
                       DATA_BUFFER_MULTIPLIER,
                       DATA_NUM_PARALLEL_CALLS)

# `iterator_utils_3` is used in settings where the source
# datasets are pre-computed sequence embeddings, e.g.
# ELMo representations that are computed aforeheard and stored
# to disk. Thus for testing purpose, we will use the same data.
#
# In particular, we need these files:
#   data.labels, data.label_vocab
#   data.sequence_1, data.sequence_2,
#   data.sequence_1.elmo.hdf5, data.sequence_2.elmo.hdf5
#
# where *.elmo.hdf5 are HDF5 files of ELMo representations of
# the corresponding data. The hdf5.get(i) returns ELMo representation
# for i-th sequence, and hdf5.get(sentence_to_index) returns a dict
# mapping sentence to index. This data format comes from Allennlp.ELMo.
# Unfortunately the dataset is too big to be saved in this repo, so it's
# kept separate.

TEST_FNAME = "/nlp/han/DLResearch/DATA/TextClassifDataELMO/RTE-30K/train"


def get_data_batch(fname, batch_size=32, random_seed=3):
    # keep the variable naming the same
    val_file = fname
    val_batch_size = batch_size

    def _data_generator(fname):
        h5py_data = h5py.File(fname + ".elmo.hdf5", "r")
        # Iterating over `.keys()` and find length or
        # using `len(h5py_data)` in large datasets
        # prohibitative, instead, `sentence_to_index`
        # is much faster to get. We can also use
        # while-loop, and break when output is None.
        # Speed: When dataset is small (e.g. RTE/MRPC)
        # using `len()` is roughly 10-20X faster than my
        # approach (40ms vs. 2ms). When dataset is large (e.g. QNLI)
        # my approach takes 500ms versus 2min for `len()`.
        sentence_to_index = eval(  # "{S1: Index1, S2: Index2,...}"
            h5py_data.get("sentence_to_index").value[0])
        # Instead of using `len(sentence_to_index.keys())`
        # use `max(sentence_to_index.values)` because there
        # can be duplicate sentences, and thus using `len()`
        # will lead to smaller `num_elements` than supposed to be.
        # `+1` because max index + 1 = length
        # Note that this might also lead to incorrect count
        # because in duplicate settings, we cannot guarantee
        # that `max(sentence_to_index.values)` will return
        # the correct length when the multiple sentences
        # are duplicate form of the sentence at the max index.
        # But this might not be a problem here because the max index
        # is usually inserted at the end, and thus will not be
        # overriden. Some tests will be used to verify this.
        num_elements = np.max([int(i) for i in sentence_to_index.values()]) + 1

        def _callable_generator():
            for i in range(num_elements):
                # [3, sequence_length, 1024]
                raw_data = h5py_data.get(str(i))
                raw_data = raw_data.value
                yield raw_data
        return _callable_generator
    
    # Note that the label.vocab file for train/val/test
    # can be different, i.e., the same label will be mapped
    # to a differnt integer with different label.vocab file
    # so ALWAYS use train.label_vocab for consistency
    # [For Test, use val_file]
    tgt_vocab_file = val_file + ".label_vocab"

    # no UNKs in target labels
    tgt_vocab_table = lookup_ops.index_table_from_file(tgt_vocab_file)

    val_src_1 = tf.data.Dataset.from_generator(
        _data_generator(val_file + ".sequence_1"),
        output_types=tf.float32,
        output_shapes=tf.TensorShape(
            [CACHED_ELMO_NUM_ELEMENTS, None, CACHED_ELMO_NUM_UNITS]))
    val_src_2 = tf.data.Dataset.from_generator(
        _data_generator(val_file + ".sequence_2"),
        output_types=tf.float32,
        output_shapes=tf.TensorShape(
            [CACHED_ELMO_NUM_ELEMENTS, None, CACHED_ELMO_NUM_UNITS]))
    val_tgt = tf.data.TextLineDataset(val_file + ".labels")
    val_batch = iterator_utils_3.get_pairwise_classification_iterator(
        src_dataset_1=val_src_1,
        src_dataset_2=val_src_2,
        tgt_dataset=val_tgt,
        tgt_vocab_table=tgt_vocab_table,
        batch_size=val_batch_size,
        random_seed=random_seed,
        src_len_axis=1,
        num_parallel_calls=DATA_NUM_PARALLEL_CALLS,
        output_buffer_size=val_batch_size * DATA_BUFFER_MULTIPLIER,
        shuffle=False,
        repeat=False)
    
    return val_batch



class DataReaderTest(tf.test.TestCase):

    def testELMOData(self):
        fname = TEST_FNAME
        data_batch = get_data_batch(fname)
        with self.test_session() as session:
            session.run(tf.tables_initializer())
            session.run(tf.global_variables_initializer())
            session.run(data_batch.initializer)

            try:
                source_1s = []
                source_2s = []
                targets = []
                while True:
                    s1, s2, tgt = session.run([
                        data_batch.source_1,
                        data_batch.source_2,
                        data_batch.target])
                    
                    source_1s.extend([
                        x.squeeze() for x in
                        np.split(s1, s1.shape[0], axis=0)])
                    source_2s.extend([
                        x.squeeze() for x in
                        np.split(s2, s2.shape[0], axis=0)])
                    targets.extend(tgt.tolist())

            except tf.errors.OutOfRangeError:
                print("DONE ", len(source_1s), len(source_2s), len(targets))

        labels = misc_utils.read_text_file_utf8(fname + ".labels")
        label_vocab = misc_utils.read_text_file_utf8(fname + ".label_vocab")

        sequence_1 = misc_utils.read_text_file_utf8(fname + ".sequence_1")
        sequence_2 = misc_utils.read_text_file_utf8(fname + ".sequence_2")

        sequence_1_h5py_data = h5py.File(fname + ".sequence_1.elmo.hdf5", "r")
        sequence_2_h5py_data = h5py.File(fname + ".sequence_2.elmo.hdf5", "r")

        sequence_1_sentence_to_index = (
            sequence_1_h5py_data.get("sentence_to_index").value)
        # assert sequence_1_sentence_to_index.shape == (1,)
        self.assertEqual(sequence_1_sentence_to_index.shape, (1,))
        sequence_1_sent2idx = eval(sequence_1_sentence_to_index[0])

        sequence_2_sentence_to_index = (
            sequence_2_h5py_data.get("sentence_to_index").value)
        # assert sequence_2_sentence_to_index.shape == (1,)
        self.assertEqual(sequence_2_sentence_to_index.shape, (1,))
        sequence_2_sent2idx = eval(sequence_2_sentence_to_index[0])

        # Ideally, `sequence_?_indices` should be [0, 1, ..., N]
        # However because of the duplicates, the numbers can be
        # [0, 99, 1, 2, ..., N] where the 99 in the second element
        # means the second string also appears in 100-th element
        sequence_1_indices = [int(sequence_1_sent2idx[s]) for s in sequence_1]
        sequence_2_indices = [int(sequence_2_sent2idx[s]) for s in sequence_2]
        # Thus use the `sequence_?_indices` to query
        sequence_1_embs = [sequence_1_h5py_data.get(str(i)).value
                           for i in sequence_1_indices]
        sequence_2_embs = [sequence_2_h5py_data.get(str(i)).value
                           for i in sequence_2_indices]

        # Use `sequence_1_indices` instead of `range(len(sequence_1_embs))`
        # in particular, even though the duplicates in the indices
        # mean that index X and idex J will refer to the same string,
        # the non-determinism in ELMO means that they will receive slightly
        # different embedding (maybe?).
        for idx in sequence_1_indices:
            computed_embs = source_1s[idx]
            # `np.nonzero(computed_embs)` is zero-padded,
            # remove the padding. `np.nonzero` returns
            # (index_0, index_1, ...) over ndims of the flattened
            # inputs. Since the zeros are only padded on sequence
            # length dim, so take [1] of the results, and unique them
            mask = np.unique(np.nonzero(computed_embs)[1])
            computed_embs = computed_embs[:, mask, :]
            
            # assert (sequence_1_embs[idx] == computed_embs).all()
            self.assertAllEqual(sequence_1_embs[idx], computed_embs)

        for idx in sequence_2_indices:
            computed_embs = source_2s[idx]
            # `np.nonzero(computed_embs)` is zero-padded,
            # remove the padding. `np.nonzero` returns
            # (index_0, index_1, ...) over ndims of the flattened
            # inputs. Since the zeros are only padded on sequence
            # length dim, so take [1] of the results, and unique them
            mask = np.unique(np.nonzero(computed_embs)[1])
            computed_embs = computed_embs[:, mask, :]
            
            # assert (sequence_2_embs[idx] == computed_embs).all()
            self.assertAllEqual(sequence_2_embs[idx], computed_embs)

        # assert targets == [label_vocab.index(l) for l in labels]
        self.assertEqual(targets, [label_vocab.index(l) for l in labels])


if __name__ == "__main__":
    tf.test.main()
