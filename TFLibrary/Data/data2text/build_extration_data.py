from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import json
import h5py
import codecs
import numpy as np
from collections import Counter
from TFLibrary.Data.data2text import utils
from TFLibrary.Data.data2text import vocabulary
from TFLibrary.Data.data2text import original_data_utils as orig_utils

TRAIN_JSON = "/Users/AlexGuo/Downloads/boxscore-data/rotowire/train.json"
VAL_JSON = "/Users/AlexGuo/Downloads/boxscore-data/rotowire/valid.json"
TEST_JSON = "/Users/AlexGuo/Downloads/boxscore-data/rotowire/valid.json"


def build_extration_data(train_json_file=TRAIN_JSON,
                         val_json_file=VAL_JSON,
                         test_json_file=TEST_JSON,
                         output_file="./IE_data"):
    print("WARNING: RE-BUILD THE DATA TO ENSURE THEY ARE THE LATEST")
    # ===========================================================
    # set_up
    # ===========================================================
    with codecs.open(train_json_file, "r", "utf-8") as f:
        train_data = json.load(f)
    with codecs.open(val_json_file, "r", "utf-8") as f:
        val_data = json.load(f)
    # not sure why these two files are the same ?
    with codecs.open(test_json_file, "r", "utf-8") as f:
        test_data = json.load(f)

    # ===========================================================
    # extract_entities_from_json
    # ===========================================================
    (all_entities,
     players,
     teams,
     cities) = utils.extract_entities_from_json(
        json_file=train_json_file, lower=False)

    # ===========================================================
    # process_candidate_rels
    # ===========================================================
    extracted = []
    # shape = [num_datasets][num_entries]
    for dataset in [train_data, val_data, test_data]:
        nugz = []
        for entry in dataset:
            _nugz = utils.process_candidate_rels(
                entry=entry,
                summary=" ".join(entry["summary"]),
                all_entities=all_entities,
                pronouns=orig_utils.prons,
                players=players,
                teams=teams,
                cities=cities)

            # use += (), this is hard to debug
            nugz += (_nugz)
        extracted.append(nugz)


    # ===========================================================
    # Vocabulary
    # ===========================================================
    word_counter = Counter()
    label_counter = Counter()
    for cand_rels in extracted[0]:
        # [0] stands for training set
        # word counter counts all tokens in a summary
        word_counter.update(cand_rels.Tokens)
        # label counter counts all Labels in relations
        label_counter.update([entry.Label for entry in cand_rels.Relations])

    word_vocab = vocabulary.Vocabulary(
        [p[0] for p in word_counter.items() if p[1] >= 2])
    label_vocab = vocabulary.Vocabulary(
        [p[0] for p in label_counter.items()])



    # ===========================================================
    # process_multilabeled_data and append_labelnums
    # ===========================================================
    (train_token_ids_list,
     train_token_lens_list,
     train_entity_dists,
     train_number_dists,
     train_label_ids_list) = utils.collect_all_features(
        extracted_features=extracted[0],
        word_vocab=word_vocab,
        label_vocab=label_vocab)

    (val_token_ids_list,
     val_token_lens_list,
     val_entity_dists,
     val_number_dists,
     val_label_ids_list) = utils.collect_all_features(
        extracted_features=extracted[1],
        word_vocab=word_vocab,
        label_vocab=label_vocab)

    (test_token_ids_list,
     test_token_lens_list,
     test_entity_dists,
     test_number_dists,
     test_label_ids_list) = utils.collect_all_features(
        extracted_features=extracted[2],
        word_vocab=word_vocab,
        label_vocab=label_vocab)


    # write vocabs
    word_vocab.save(output_file + ".word_vocab")
    label_vocab.save(output_file + ".label_vocab")
    
    # write datasets
    list2arr = lambda l: np.array(l)
    with h5py.File(output_file + ".data", "w") as f:
        f.create_dataset("train/token_ids_list",
            data=list2arr(train_token_ids_list))
        f.create_dataset("train/token_lens_list",
            data=list2arr(train_token_lens_list))
        f.create_dataset("train/entity_dists",
            data=list2arr(train_entity_dists))
        f.create_dataset("train/number_dists",
            data=list2arr(train_number_dists))
        f.create_dataset("train/label_ids_list",
            data=list2arr(train_label_ids_list))

        f.create_dataset("val/token_ids_list",
            data=list2arr(val_token_ids_list))
        f.create_dataset("val/token_lens_list",
            data=list2arr(val_token_lens_list))
        f.create_dataset("val/entity_dists",
            data=list2arr(val_entity_dists))
        f.create_dataset("val/number_dists",
            data=list2arr(val_number_dists))
        f.create_dataset("val/label_ids_list",
            data=list2arr(val_label_ids_list))

        f.create_dataset("test/token_ids_list",
            data=list2arr(test_token_ids_list))
        f.create_dataset("test/token_lens_list",
            data=list2arr(test_token_lens_list))
        f.create_dataset("test/entity_dists",
            data=list2arr(test_entity_dists))
        f.create_dataset("test/number_dists",
            data=list2arr(test_number_dists))
        f.create_dataset("test/label_ids_list",
            data=list2arr(test_label_ids_list))

    print("Finished Saving Files to ", output_file)
