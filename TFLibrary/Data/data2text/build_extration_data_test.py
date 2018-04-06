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

    for suffix in [".data", ".word_vocab", ".label_vocab",
                   ".entities", ".players", ".teams", ".cities"]:
        misc_utils.maybe_delete_file(output_file_a + suffix, check_exists=True)
    misc_utils.maybe_delete_file(output_file_b, check_exists=True)

    return train_tuples_a, train_tuples_b, val_tuples_a, val_tuples_b


def test_prepare_extraction_data_for_eval(IE_data_file,
                                          summary_files,
                                          output_file="./IE-tmp",
                                          summmary_process_fn=None):
    """
    To Test prepare_extraction_data_for_eval
    We use this function to build traning, validation, and test
    dataset, and compare them against the datasets built from
    other methods

    
    Example for Running

        from TFLibrary.Data.data2text.build_extration_data_test import test_prepare_extraction_data_for_eval
        from TFLibrary.Data.data2text.build_data import reverse_outputs as reverse_outputs_fn
        from extraction_model import multilabel_accuracy
        IE_data_file = "../../DATA/TextSummData/BoxScore/IE_data"
        summary_files = ["../../DATA/TextSummData/BoxScore/train_target.txt",
                         "../../DATA/TextSummData/BoxScore/val_target.txt",
                         "../../DATA/TextSummData/BoxScore/test_target.txt"]
        
        summmary_process_fn = multilabel_accuracy
        test_prepare_extraction_data_for_eval(
            IE_data_file=IE_data_file,
            summary_files=summary_files,
            summmary_process_fn=reverse_outputs_fn)

    """
    modes = ["train", "val", "test"]
    json_files = [build_extration_data.TRAIN_JSON,
                  build_extration_data.VAL_JSON,
                  build_extration_data.TEST_JSON]
    
    if len(json_files) != len(summary_files):
        raise ValueError("json_files and summary_files different shapes")


    h5file_expected = h5py.File(IE_data_file + ".data", "r")
    for mode, json_file, summary_file in zip(modes,
                                             json_files,
                                             summary_files):
        build_extration_data.prepare_extraction_data_for_eval(
            json_file=json_file,
            IE_data_file=IE_data_file,
            summary_file=summary_file,
            output_file=output_file,
            summmary_process_fn=summmary_process_fn)

        # read the data
        data_expected = h5file_expected[mode]
        
        h5file_actual = h5py.File(output_file + ".data", "r")
        data_actual = h5file_actual["evaluation"]


        tuples_expected = (data_expected["token_ids_list"][:],
                           data_expected["token_lens_list"][:],
                           data_expected["entity_dists"][:],
                           data_expected["number_dists"][:],
                           data_expected["label_ids_list"][:])

        tuples_actual = (  data_actual["token_ids_list"][:],
                           data_actual["token_lens_list"][:],
                           data_actual["entity_dists"][:],
                           data_actual["number_dists"][:],
                           data_actual["label_ids_list"][:])

        print("Testing Mode ", mode)
        for (t1, t2) in zip(tuples_expected, tuples_actual):
            print((t1 == t2).all(), "\t", t1.shape, t2.shape)
        misc_utils.maybe_delete_file(output_file + ".data", check_exists=True)
