from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import h5py
import json
import copy
import codecs
import numpy as np
from nltk import sent_tokenize
from collections import namedtuple
from collections import defaultdict

from TFLibrary.Data.data2text.original_data_utils import prons as PRONS
from TFLibrary.Data.data2text.original_data_utils import extract_numbers, extract_entities, get_rels


EntityEntry = namedtuple(
    "EntityEntry",
    ("WindowLeft", "WindowRight", "Token", "IsPronoun"))
NumberEntry = namedtuple(
    "NumberEntry",
    ("WindowLeft", "WindowRight", "Token"))
RelationEntry = namedtuple(
    "RelationEntry",
    ("Entity", "Number", "Label", "PlayerID_or_TeamIsHome"))
CandidateRelations = namedtuple(
    "CandidateRelations",
    ("Tokens", "Relations"))


def _flatten(L):
    return [item for sublist in L for item in sublist]


def _process_compound_words(entities):
    UNWANTED_TOKENS = [
        "II", "III", "Jr.", "Jr",
        "II".lower(), "III".lower(),
        "Jr.".lower(), "Jr".lower()]

    additional_entities = []
    for entity in entities:
        # split phrases into tokens
        tokens = entity.split()
        # compound words
        if len(tokens) > 1:
            # append "-".join tokens
            # because I did this processing for inputs
            # additional_entities.append("-".join(tokens))
            for token in tokens:
                # remove some redundant words
                # append single tokens
                if len(token) > 1 and token not in UNWANTED_TOKENS:
                    additional_entities.append(token)

    return additional_entities


def extract_entities_from_json(raw_data):
    teams = []
    cities = []
    players = []
    for entry in raw_data:
        teams += [
            # home
            entry["home_name"],
            entry["home_line"]["TEAM-NAME"],
            " ".join([entry["home_city"], entry["home_name"]]),
            " ".join([entry["home_city"], entry["home_line"]["TEAM-NAME"]]),
            # visiting
            entry["vis_name"],
            entry["vis_line"]["TEAM-NAME"],
            " ".join([entry["vis_city"], entry["vis_name"]]),
            " ".join([entry["vis_city"], entry["vis_line"]["TEAM-NAME"]])]

        # special case for this
        if entry["vis_city"] == "Los Angeles":
            teams.append("LA" + entry["vis_name"])
        if entry["home_city"] == "Los Angeles":
            teams.append("LA" + entry["home_name"])

        # sometimes team_city is different
        cities += (
            [entry["home_city"], entry["vis_city"]] +
            list(entry["box_score"]["TEAM_CITY"].values()))

        players += entry["box_score"]["PLAYER_NAME"].values()

    teams += _process_compound_words(teams)
    cities += _process_compound_words(cities)
    players += _process_compound_words(players)

    teams = set(teams)
    cities = set(cities)
    players = set(players)

    all_entities = players | teams | cities
    return all_entities, players, teams, cities


def process_candidate_rels(entry, summary,
                           all_entities, pronouns,
                           players, teams, cities):
    """return form (sentence_tokens, [rels]) to candrels"""
    candidate_relations = []
    sentences = sent_tokenize(summary)
    for sentence in sentences:
        tokens = sentence.split()
        numbers = extract_numbers(tokens)
        entities = extract_entities(tokens, all_entities, pronouns)

        # make them named tuples
        numbers = [NumberEntry(*num) for num in numbers]
        entities = [EntityEntry(*ent) for ent in entities]

        relations = get_rels(entry, entities, numbers, players, teams, cities)
        # make them named tuples
        relations = [RelationEntry(*rel) for rel in relations]

        if len(relations) > 0:
            candidate_relations.append(
                CandidateRelations(Tokens=tokens, Relations=relations))

    return candidate_relations


def tokens_padding(tokens, max_length, pad_token="PAD"):
    tokens_len = len(tokens)
    pad_length = max_length - tokens_len
    if pad_length < 0:
        raise ValueError("Pad Length < 0: ", tokens)

    tokens.extend([pad_token for _ in range(pad_length)])
    return tokens, tokens_len


def process_multilabeled_data(candidate_relation,
                              tokens_max_length,
                              token_look_up_fn,
                              label_look_up_fn):

    if not len(candidate_relation) == 2:
        raise ValueError("candidate_relation has more than 2 fields")
    if not isinstance(candidate_relation, CandidateRelations):
        raise TypeError

    # relationships:
    # Entity + Number --> Feature
    unique_relations = defaultdict(list)
    for relation in candidate_relation.Relations:
        if not isinstance(relation, RelationEntry):
            raise TypeError
        # dict[key1, key2] = list(val1, val2, ...)
        unique_relations[relation.Entity,
                         relation.Number].append(relation.Label)

    entity_dists = []
    number_dists = []
    token_ids_list = []
    token_lens_list = []
    label_ids_list = []
    tokens = copy.deepcopy(candidate_relation.Tokens)
    padded_tokens, tokens_len = tokens_padding(tokens, tokens_max_length)
    token_ids = list(map(token_look_up_fn, padded_tokens))
    for entnum_pair, labels in unique_relations.items():

        (entity, number) = entnum_pair
        if not isinstance(entity, EntityEntry):
            raise TypeError
        if not isinstance(number, NumberEntry):
            raise TypeError

        token_ids_list.append(token_ids)
        token_lens_list.append(tokens_len)

        label_ids = list(map(label_look_up_fn, labels))
        label_ids_list.append(label_ids)

        entity_dist = [time_idx - entity.WindowLeft
                       if time_idx < entity.WindowLeft
                       else time_idx - entity.WindowRight + 1
                       if time_idx >= entity.WindowRight
                       else 0 for time_idx in range(tokens_max_length)]

        number_dist = [time_idx - number.WindowLeft
                       if time_idx < number.WindowLeft
                       else time_idx - number.WindowRight + 1
                       if time_idx >= number.WindowRight
                       else 0 for time_idx in range(tokens_max_length)]

        entity_dists.append(entity_dist)
        number_dists.append(number_dist)

    return (token_ids_list, token_lens_list,
            entity_dists, number_dists, label_ids_list)


def append_labelnums(all_label_ids_list, pad_id=-1):
    # make sure don't mess up references
    new_label_ids_list = copy.deepcopy(all_label_ids_list)
    # number of labels in every pair
    label_nums = [len(lidlist) for lidlist in all_label_ids_list]
    max_num_labels = max(label_nums)
    print("max num labels ", max_num_labels)

    # append number of labels to labels
    for i, lidlist in enumerate(new_label_ids_list):
        # do the padding and append number of Labels
        # at the end of the sequence, e.g. :
        # from [Label_1, Label_2, ..., Label_N]
        # into [Label_1, Label_2, ..., Label_N, -1, -1, ... N]
        pad_length = max_num_labels - len(lidlist)
        lidlist.extend([pad_id for _ in range(pad_length)])
        lidlist.append(label_nums[i])

    return new_label_ids_list


def collect_all_features(extracted_features, word_vocab, label_vocab):
    all_token_ids_list = []
    all_token_lens_list = []
    all_entity_dists = []
    all_number_dists = []
    all_label_ids_list = []
    # for training set
    max_len = max([len(cand_rels.Tokens)
        for cand_rels in extracted_features])
    for cand_rels in extracted_features:
        (token_ids_list,
         token_lens_list,
         entity_dists,
         number_dists,
         label_ids_list) = process_multilabeled_data(
            candidate_relation=cand_rels,
            tokens_max_length=max_len,
            token_look_up_fn=lambda token: (
                -1 if token == "PAD"
                else word_vocab.word_to_id(token)),
            label_look_up_fn=label_vocab.word_to_id)

        all_token_ids_list += token_ids_list
        all_token_lens_list += token_lens_list
        all_entity_dists += entity_dists
        all_number_dists += number_dists
        all_label_ids_list += label_ids_list

    new_label_ids_list = append_labelnums(all_label_ids_list)

    return (all_token_ids_list,
            all_token_lens_list,
            all_entity_dists,
            all_number_dists,
            new_label_ids_list)


def tile_dataset(token_ids_list,
                 token_lens_list,
                 entity_dists,
                 number_dists,
                 label_ids_list,
                 tile=True,
                 expand_label=False):
    """
    Original label_ids_list contains multiple ground-truth
    labels for a given input pairs. Here we tile datasets
    so that each input-output pair has single target
    """
    if not token_ids_list.shape[0] == entity_dists.shape[0]:
        raise AssertionError
    if not token_ids_list.shape[0] == token_lens_list.shape[0]:
        raise AssertionError
    if not token_ids_list.shape[0] == number_dists.shape[0]:
        raise AssertionError
    if not token_ids_list.shape[0] == label_ids_list.shape[0]:
        raise AssertionError
    
    tiled_tokens = []
    tiled_token_lens = []
    tiled_entity_dists = []
    tiled_number_dists = []
    tiled_label_ids = []
    for row_idx in range(token_ids_list.shape[0]):
        # the last number in label_ids_list
        # tells the number of labels with the same inputs
        token_ids = token_ids_list[row_idx, :]
        token_lens = token_lens_list[row_idx]
        entity_dist = entity_dists[row_idx, :]
        number_dist = number_dists[row_idx, :]
        label_ids = label_ids_list[row_idx, :]
        
        if not tile:
            tiled_tokens.append(token_ids)
            tiled_token_lens.append(token_lens)
            tiled_entity_dists.append(entity_dist)
            tiled_number_dists.append(number_dist)
            tiled_label_ids.append(label_ids[:-1])

        else:
            # the last element contains the number of labels
            for label_idx in range(label_ids[-1]):
                if expand_label:
                    _label_ids = np.expand_dims(label_ids[label_idx], axis=-1)
                else:
                    _label_ids = label_ids[label_idx]

                tiled_tokens.append(token_ids)
                tiled_token_lens.append(token_lens)
                tiled_entity_dists.append(entity_dist)
                tiled_number_dists.append(number_dist)
                tiled_label_ids.append(_label_ids)
    
    return (tiled_tokens,
            tiled_token_lens,
            tiled_entity_dists,
            tiled_number_dists,
            tiled_label_ids)


def prepare_generated_data(train_json_file,
                           eval_json_file,
                           gen_file,
                           output_file=None,
                           vocabulary=None,
                           token_dict=None,
                           label_dict=None):

    raise NotImplementedError("This function is depreciated")

    # read all relevant files
    with codecs.open(gen_file, "r", "utf-8") as f:
        generated_summaries = f.readlines()
    with codecs.open(train_json_file, "r", "utf-8") as f:
        train_data = json.load(f)
    with codecs.open(eval_json_file, "r", "utf-8") as f:
        eval_data = json.load(f)
    if len(eval_data) != len(generated_summaries):
        raise ValueError

    # Step 1: extract all the entities etc from train data
    all_entities, players, teams, cities = extract_entities_from_json(
        raw_data=train_data)

    # Step 2: extract all the candidate relationship pairs
    candidate_relations = []  # to hold (sentence_tokens, [rels]) tuples
    sent_reset_indices = {0}  # sentence indices where a box/story is reset
    for i, entry in enumerate(eval_data):
        summary = generated_summaries[i]
        candidate_relation = process_candidate_rels(
            entry=entry, summary=summary,
            all_entities=all_entities, pronouns=PRONS,
            players=players, teams=teams, cities=cities)
        candidate_relations += (candidate_relation)
        sent_reset_indices.add(len(candidate_relations))

    # Step 3 process multi-labels and padding
    if vocabulary is not None:
        token_look_up_fn = lambda word: vocab.word2id(word)
        label_look_up_fn = lambda label: vocab.word2id(label)

    # recreate vocab and labeldict
    elif token_dict is not None and label_dict is not None:
        vocab = {}
        with codecs.open(token_dict, "r", "utf-8") as f:
            for line in f:
                pieces = line.strip().split()
                vocab[pieces[0]] = int(pieces[1])
        labeldict = {}
        with codecs.open(label_dict, "r", "utf-8") as f:
            for line in f:
                pieces = line.strip().split()
                labeldict[pieces[0]] = int(pieces[1])

        def token_look_up_fn(word):
            if word == "PAD":
                return -1
            if word in vocab:
                return vocab[word]
            else:
                return vocab["UNK"]

        label_look_up_fn = lambda label: labeldict[label]

    else:
        # only used in debugging mode
        token_look_up_fn = lambda word: word
        label_look_up_fn = lambda label: label

    max_len = max((len(cr[0]) for cr in candidate_relations))
    all_token_ids_list = []
    all_token_lens_list = []
    all_entity_dists = []
    all_number_dists = []
    all_label_ids_list = []
    rel_reset_indices = []
    for t, cand_relation in enumerate(candidate_relations):
        # then last rel is the last of its box
        if t in sent_reset_indices:
            if len(all_token_ids_list) != len(all_token_lens_list):
                raise ValueError
            rel_reset_indices.append(len(all_token_ids_list))

        (token_ids_list,
         token_lens_list,
         entity_dists,
         number_dists,
         label_ids_list) = process_multilabeled_data(
            candidate_relation=cand_relation,
            tokens_max_length=max_len,
            token_look_up_fn=token_look_up_fn,
            label_look_up_fn=label_look_up_fn)

        all_token_ids_list += token_ids_list
        all_token_lens_list += token_lens_list
        all_entity_dists += entity_dists
        all_number_dists += number_dists
        all_label_ids_list += label_ids_list

    # Step 4, process label numbers and padding
    all_label_ids_list = append_labelnums(all_label_ids_list)

    # Step 5: Write to H5
    print(len(all_label_ids_list), " prediction examples")

    if output_file is not None:
        h5fi = h5py.File(output_file, "w")
        h5fi["valsents"] = np.array(all_token_ids_list, dtype=int)
        h5fi["vallens"] = np.array(all_token_lens_list, dtype=int)
        h5fi["valentdists"] = np.array(all_entity_dists, dtype=int)
        h5fi["valnumdists"] = np.array(all_number_dists, dtype=int)
        h5fi["vallabels"] = np.array(all_label_ids_list, dtype=int)
        h5fi["boxrestartidxs"] = np.array(  # 1-indexed in torch
            np.array(rel_reset_indices) + 1, dtype=int)
        h5fi.close()

    return (all_token_ids_list,
            all_token_lens_list,
            all_entity_dists,
            all_number_dists,
            all_label_ids_list,
            rel_reset_indices)
