from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import h5py
from TFLibrary.utils import misc_utils
from TFLibrary.Data.data2text import vocabulary
from TFLibrary.Data.data2text import build_extration_data
from TFLibrary.Data.data2text import original_data_utils as orig_utils


def test(output_file_a="./IE-data-a",
         output_file_b="./IE-data-b"):
    # build the data
    build_extration_data.build_extration_data(
        train_json_file=build_extration_data.TRAIN_JSON,
        val_json_file=build_extration_data.VAL_JSON,
        test_json_file=build_extration_data.TEST_JSON,
        output_file=output_file_a)

    # extract vocabularies
    word_vocab = vocabulary.Vocabulary([None])
    label_vocab = vocabulary.Vocabulary([None])
    word_vocab.load(output_file_a + ".word_vocab")
    label_vocab.load(output_file_a + ".label_vocab")

    # use above vocabularies for consistency
    orig_utils.save_full_sent_data(
        outfile=output_file_b,
        path=build_extration_data.DATA_BASE_DIR,
        multilabel_train=True,
        replacement_vocab=word_vocab.vocab,
        replacement_labeldict=label_vocab.vocab)
    
    # read the data
    h5file_a = h5py.File(output_file_a + ".data", "r")
    h5file_b = h5py.File(output_file_b, "r")

    # train
    train_data_a = h5file_a["train"]
    train_tuples_a = (train_data_a["token_ids_list"][:],
                      train_data_a["token_lens_list"][:],
                      train_data_a["entity_dists"][:],
                      train_data_a["number_dists"][:],
                      train_data_a["label_ids_list"][:])

    train_data_b = h5file_b
    train_tuples_b = (train_data_b["trsents"][:],
                      train_data_b["trlens"][:],
                      train_data_b["trentdists"][:],
                      train_data_b["trnumdists"][:],
                      train_data_b["trlabels"][:])

    # validation
    val_data_a = h5file_a["val"]
    val_tuples_a = (val_data_a["token_ids_list"][:],
                    val_data_a["token_lens_list"][:],
                    val_data_a["entity_dists"][:],
                    val_data_a["number_dists"][:],
                    val_data_a["label_ids_list"][:])

    val_data_b = h5file_b
    val_tuples_b = (val_data_b["valsents"][:],
                    val_data_b["vallens"][:],
                    val_data_b["valentdists"][:],
                    val_data_b["valnumdists"][:],
                    val_data_b["vallabels"][:])

    for (tda, tdb,
         vda, vdb) in zip(train_tuples_a, train_tuples_b,
                          val_tuples_a, val_tuples_b):
        print(tda.shape, vda.shape)
        print((tda == tdb).all(), (vda == vdb).all(), tda != vda)

    for suffix in [".data", ".word_vocab", ".label_vocab"]:
        misc_utils.maybe_delete_file(output_file_a + suffix, check_exists=True)
    misc_utils.maybe_delete_file(output_file_b, check_exists=True)

    return train_tuples_a, train_tuples_b, val_tuples_a, val_tuples_b
