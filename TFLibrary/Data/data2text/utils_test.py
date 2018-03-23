"""
Test against original implementations
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import copy
import json
import codecs
from collections import Counter
from TFLibrary.utils import test_utils
from TFLibrary.Data.data2text import utils
from TFLibrary.Data.data2text import original_data_utils as orig_utils


def test_components():
    # ===========================================================
    # set_up
    # ===========================================================
    json_file = "/Users/AlexGuo/Downloads/boxscore-data/rotowire/train.json"
    val_json_file = "/Users/AlexGuo/Downloads/boxscore-data/rotowire/valid.json"
    with codecs.open(json_file, "r", "utf-8") as f:
        train_data = json.load(f)
    with codecs.open(val_json_file, "r", "utf-8") as f:
        val_data = json.load(f)
    # not sure why these two files are the same ?
    with codecs.open(val_json_file, "r", "utf-8") as f:
        test_data = json.load(f)

    # ===========================================================
    # extract_entities_from_json
    # ===========================================================
    # my implementation

    (actual_all_entities,
     actual_players,
     actual_teams,
     actual_cities) = utils.extract_entities_from_json(json_file, lower=False)

    # original implementation
    (expected_all_ents,
     expected_players,
     expected_teams,
     expected_cities) = orig_utils.get_ents(train_data)


    test_utils.check_set_equality(actual_all_entities, expected_all_ents)
    test_utils.check_set_equality(actual_players, expected_players)
    test_utils.check_set_equality(actual_teams, expected_teams)
    test_utils.check_set_equality(actual_cities, actual_cities)


    # ===========================================================
    # test_process_candidate_rels
    # ===========================================================
    # my implementation
    # shape = [num_datasets][num_entries]
    actual_extracted = []
    for dataset in [train_data, val_data]:
        actual_nugz = []
        for entry in dataset:
            _nugz = utils.process_candidate_rels(
                entry=entry,
                summary=" ".join(entry["summary"]),
                all_entities=actual_all_entities,
                pronouns=orig_utils.prons,
                players=actual_players,
                teams=actual_teams,
                cities=actual_cities)

            # use += (), this is hard to debug
            actual_nugz += (_nugz)
        actual_extracted.append(actual_nugz)


    # original implementation
    expected_extracted_stuff = []
    datasets = [train_data, val_data, test_data]
    for dataset in datasets:
        expected_nugz = []
        for i, entry in enumerate(dataset):
            summ = " ".join(entry['summary'])
            orig_utils.append_candidate_rels(
                entry, summ,
                expected_all_ents,
                orig_utils.prons,
                expected_players,
                expected_teams,
                expected_cities,
                expected_nugz)

        expected_extracted_stuff.append(expected_nugz)


    # recursively check for equalities
    for extracted_a, extracted_b in zip(
            actual_extracted, expected_extracted_stuff):
        if len(extracted_a) != len(extracted_b):
            raise AssertionError("number of extratced pairs are different")

        # for all relations in each dataset
        for relations_a, relations_b in zip(extracted_a, extracted_b):
            if not isinstance(relations_a, utils.CandidateRelations):
                raise TypeError
            if len(relations_b) != 2:
                raise AssertionError
            if relations_a.Tokens != relations_b[0]:
                raise AssertionError("Tokens are different")

            # relations
            if len(relations_a.Relations) != len(relations_b[1]):
                raise AssertionError("Number of relations are different")

            # for each relationship pair
            for rel_a, rel_b in zip(relations_a.Relations, relations_b[1]):
                if not isinstance(rel_a, utils.RelationEntry):
                    raise TypeError
                if len(rel_b) != 4:
                    raise AssertionError

                if tuple(rel_a.Entity) != rel_b[0]:
                    raise AssertionError("EntityEntry are different")
                if tuple(rel_a.Number) != rel_b[1]:
                    raise AssertionError("NumberEntry are different")
                if rel_a.Label != rel_b[2]:
                    raise AssertionError("Label are different")
                if rel_a.PlayerID_or_TeamIsHome != rel_b[3]:
                    raise AssertionError("PlayerID_or_TeamIsHome are different")


    # ===========================================================
    # test Vocabulary
    # ===========================================================

    # my implementation
    word_counter = Counter()
    label_counter = Counter()
    for cand_rels in actual_extracted[0]:
        # [0] stands for training set
        # word counter counts all tokens in a summary
        word_counter.update(cand_rels.Tokens)
        # label counter counts all Labels in relations
        label_counter.update([entry.Label for entry in cand_rels.Relations])

    actual_word_vocab = utils.Vocabulary(
        [p[0] for p in word_counter.items() if p[1] >= 2])
    actual_label_vocab = utils.Vocabulary(
        [p[0] for p in label_counter.items()])


    # original implementation
    word_counter = Counter()
    [word_counter.update(tup[0]) for tup in expected_extracted_stuff[0]]
    # CHANGED ###################
    _word_counter = copy.deepcopy(word_counter)
    for k in _word_counter.keys():
    # CHANGED ###################
        if word_counter[k] < 2:
            del word_counter[k]  # will replace w/ unk
    word_counter["UNK"] = 1
    expected_vocab = dict(((wrd, i + 1)
        for i, wrd in enumerate(word_counter.keys())))
    labelset = set()
    [labelset.update([rel[2] for rel in tup[1]])
     for tup in expected_extracted_stuff[0]]
    expected_labeldict = dict(((label, i + 1)
        for i, label in enumerate(labelset)))


    vocab_passed = (set(expected_vocab.keys()) ==
                    set(actual_word_vocab.vocab.keys()))

    label_dict_passed = (
        set([k for k in expected_labeldict.keys()]) ==
        set([k for k in actual_label_vocab.vocab.keys() if k != "UNK"]))

    print(vocab_passed, label_dict_passed)


    # ===========================================================
    # test process_multilabeled_data and append_labelnums
    # ===========================================================
    # my implementation
    (actual_train_token_ids_list,
     actual_train_token_lens_list,
     actual_train_entity_dists,
     actual_train_number_dists,
     actual_train_label_ids_list) = utils.collect_all_features(
        extracted_features=actual_extracted[0],
        word_vocab=actual_word_vocab,
        label_vocab=actual_label_vocab)

    (actual_val_token_ids_list,
     actual_val_token_lens_list,
     actual_val_entity_dists,
     actual_val_number_dists,
     actual_val_label_ids_list) = utils.collect_all_features(
        extracted_features=actual_extracted[1],
        word_vocab=actual_word_vocab,
        label_vocab=actual_label_vocab)



    # original implementation
    # save stuff
    trsents, trlens, trentdists, trnumdists, trlabels = [], [], [], [], []
    valsents, vallens, valentdists, valnumdists, vallabels = [], [], [], [], []
    # testsents, testlens, testentdists, testnumdists, testlabels = [], [], [], [], []

    max_trlen = max((len(tup[0]) for tup in expected_extracted_stuff[0]))
    # print("max tr sentence length:", max_trlen)
    # do training data
    for tup in expected_extracted_stuff[0]:
        orig_utils.append_multilabeled_data(
            tup, trsents, trlens, trentdists, trnumdists, trlabels,
            # use my own dictionaries
            # expected_vocab, expected_labeldict, max_trlen)
            actual_word_vocab.vocab, actual_label_vocab.vocab, max_trlen)

    orig_utils.append_labelnums(trlabels)

    # do val, which we also consider multilabel
    max_vallen = max((len(tup[0]) for tup in expected_extracted_stuff[1]))
    for tup in expected_extracted_stuff[1]:
        #append_to_data(tup, valsents, vallens, valentdists, valnumdists, vallabels, vocab, labeldict, max_len)
        orig_utils.append_multilabeled_data(
            tup, valsents, vallens, valentdists, valnumdists, vallabels,
            # use my own dictionaries
            # vocab, labeldict, max_vallen)
            actual_word_vocab.vocab, actual_label_vocab.vocab, max_vallen)

    orig_utils.append_labelnums(vallabels)



    print(actual_train_token_ids_list == trsents,
          actual_train_token_lens_list == trlens,
          actual_train_entity_dists == trentdists,
          actual_train_number_dists == trnumdists,
          actual_train_label_ids_list == trlabels)

    print(actual_val_token_ids_list == valsents,
          actual_val_token_lens_list == vallens,
          actual_val_entity_dists == valentdists,
          actual_val_number_dists == valnumdists,
          actual_val_label_ids_list == vallabels)
