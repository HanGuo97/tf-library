"""
Test against original implementations
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import json
import codecs
from TFLibrary.utils import test_utils
from TFLibrary.Data.data2text import data_utils as utils
from TFLibrary.Data.data2text import original_data_utils as orig_utils


def test(json_file):
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
