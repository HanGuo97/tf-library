from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from collections import Counter
import pandas as pd
import numpy as np
from TFLibrary.utils import misc_utils


def process_values(example):
    """
    Processing the values for a given example

    Processing Player Data and Team Data
    1. remove PlayerName and TeamName attributes because
        we will create a separate column for this
    2. for the Team Data, reorder the columns so that
        it will produce consistent orders with dict.keys()
        This is mainly for debugging
    3. remove the column names, and replace them with numbers
        so that I can concat PlayerData and TeamData
    4. Join two DataFrames of PlayerData and TeamData, and
        replace "na" with "N/A"

    """
    # Player Data and # Team Data
    box_score = pd.DataFrame.from_records(example["box_score"])
    team = pd.DataFrame.from_dict([example["home_line"],
                                   example["vis_line"]])
    
    team = team.drop("TEAM-NAME", axis=1)
    box_score = box_score.drop("PLAYER_NAME", axis=1)


    if not (box_score.index.map(int) ==
            [int(x) for x in box_score.index]).all():
        # re-ordering (depreciated, not used)
        # unnecessary, and I am lazy anyway :)
        # make sure when mapping to integers
        # the order doesn't change
        raise AssertionError("BUG")
    
    # re-order columns to be consistent with dict.keys()
    reorder = [x for x in example["home_line"].keys() if x != "TEAM-NAME"]
    team = team[reorder]

    # make their column-names equal
    # by replacing names with actual numbers
    # (so pandas will auto-align and do padding when necessary)
    # when doing DataFrame.append, only values from corresponding
    # columns will be appended, the rest will be padded with "N/A"

    # Note that the columns names for TeamData are selected
    # so that it is essentially equivalent to pre-pad
    team_attr = team.columns
    player_attr = box_score.columns
    
    box_score.columns = range(box_score.shape[1])
    team.columns = range(len(player_attr) - team.shape[1], len(player_attr))

    # do the joining and padding
    # and fill NaN with N/A
    processed_values = box_score.append(
        team, ignore_index=True).fillna(value="N/A")

    return processed_values



def process_entities(example):
    """
    Processing the Entities for a given example

    Processing Player Data and Team Data
    1. Extract all PlayerNames and TeamNames column
    2. Note that for each Value we will assign it
        an according entity, i.e. #Attrbutes = #Entitues
        thus we will repeat each entity by len(attributes)

    """
    box_score = pd.DataFrame.from_records(example["box_score"])
    team = pd.DataFrame.from_dict([example["home_line"],
                                   example["vis_line"]])

    player_names = box_score["PLAYER_NAME"].values.tolist()
    team_names = team["TEAM-NAME"].values.tolist()

    # Here we calculate the number of entity repetitions
    # we need to do. We "-1" because the name column is dropped
    player_num_attr = box_score.shape[1] - 1
    team_num_attr = team.shape[1] - 1
    
    # Note that since there are less TeamAttributes
    # than PlayerAttributes, we will pad TeamAttributes
    # so that they have equal lengths
    
    # team-names will be pre-padded for length num_attr_diff
    num_attr_diff = player_num_attr - team_num_attr
    print("player_num_attr ", player_num_attr,
          "\nteam_num_attr ", team_num_attr,
          "\nnum_attr_diff ", num_attr_diff)

    # list of player names repeated over num_player_attr (i.e. num-columns)
    # and a list of team names repeated over same length with pre-padding
    entities = []
    for name in player_names:
        entities.append([name] * player_num_attr)
    for name in team_names:
        entities.append(["N/A"] * num_attr_diff + [name] * team_num_attr)

    return entities, num_attr_diff


def process_types(example, num_attr_diff):
    """
    Processing the Types or Relationships for a given example
    the Types = the relationship between each value and entity
    and is just the column name

    Processing Player Data and Team Data
    1. Extract all PlayerData and TeamData column
    2. Pad TeamAttributes to the same length with PlayerAttributes

    """
    box_score = pd.DataFrame.from_records(example["box_score"])
    team = pd.DataFrame.from_dict([example["home_line"],
                                   example["vis_line"]])

    num_players = box_score.shape[0]
    num_teams = team.shape[0]
    if not num_teams == 2:
        raise AssertionError("Number of Teams can't succeed 2")
    print("num_players ", num_players, "num_teams ", num_teams)

    # Note that the following equality holds
    # set(team.columns.tolist()) == set(example["home_line"].keys())
    # but for team_attr, use the order from dict.keys() because this has
    # the consistent order with my another codes used for testing
    if example["home_line"].keys() == team.columns.tolist():
        raise AssertionError("Order from dict.keys() == order from .tolist()"
                             " This is not a serious issue, but please check it")

    # and remove NAME attributes because these are represented as r.e already
    player_attrs = [x for x in box_score.columns.tolist() if x != "PLAYER_NAME"]
    # team_attrs = [x for x in team.columns.tolist() if x != "TEAM-NAME"]
    team_attrs = [x for x in example["home_line"].keys() if x != "TEAM-NAME"]
    # pre-padding
    team_attrs = ["N/A"] * num_attr_diff + team_attrs


    # the actual data looks like this
    #   Score1, Score2, Score3, Score4, Score5, ...
    #   Player1, Player1, Player2, Player2, Player3, ...
    #   Attr1, Attr2, Attr1, Attr2, Attr1, ...

    # tile these attributes to num_players and num_teams
    # these are all lists of list of shape
    # [num_entities, num_attributes]
    player_attrs = [player_attrs for _ in range(num_players)]
    team_attrs = [team_attrs for _ in range(num_teams)]
    
    # [num_all_entities, num_attributes]
    joint_attrs = player_attrs + team_attrs

    return joint_attrs



def process_home_or_away(example):
    """
    Processing the Home or Away features

    Processing Player Data and Team Data
    1. Extract the supposed Home and Visting city name
    2. and Extract Player and Team's City
    3. Compare them and decide the home_or_away feature
        for each entity. We do this because the players
        and teams are not ordered by team or away
    4. Pad TeamAttributes to the same length with PlayerAttributes

    """

    box_score = pd.DataFrame.from_records(example["box_score"])
    team = pd.DataFrame.from_dict([example["home_line"],
                                   example["vis_line"]])

    def get_home_or_away(team_city):
        if team_city == example["home_line"]["TEAM-CITY"]:
            return "Home"
        elif team_city == example["vis_line"]["TEAM-CITY"]:
            return "Away"
        raise ValueError

    player_cities = box_score["TEAM_CITY"].tolist()
    team_cities = team["TEAM-CITY"].values.tolist()

    # calculate the number of attributes, used for
    # padding team attributes
    # -1 because of dropping the name column
    player_num_attr = box_score.shape[1] - 1
    team_num_attr = team.shape[1] - 1

    # team-home-or-aways will be pre-padded for length num_attr_dirr
    num_attr_diff = player_num_attr - team_num_attr
    print("player_num_attr ", player_num_attr,
          "\nteam_num_attr ", team_num_attr,
          "\nnum_attr_diff ", num_attr_diff)


    # list of player cities repeated over num_player_attr (i.e. num-columns)
    # and a list of team cities repeated over same length with pre-padding
    # Home is city == home_line.TEAM-CITY
    # Away is city == vis_line.TEAM-CITY
    home_or_aways = []
    for city in player_cities:
        home_or_away = get_home_or_away(city)
        home_or_aways.append([home_or_away] * player_num_attr)
    for city in team_cities:
        home_or_away = get_home_or_away(city)
        home_or_aways.append(["N/A"] * num_attr_diff + [home_or_away] * team_num_attr)

    return home_or_aways



def process_compound_words(example, records_array, join_by="[SPACE]", lower=False):
    # collecting all words in the records array
    counter = Counter()
    for row in records_array.tolist():
        counter.update(row)

    # map them to joined and lowered counterpart
    mapping_dict = {}
    for key in counter.keys():
        # this "if" is essential
        # because for names, there exist three columns
        # i.e. FullNames, FirstName, and LastName
        # thus if not stricting this dictionary
        # to compound words, when do the dict[key]
        # names will probably be mapped to
        # FirstName.lower(), LastName.lower()
        # rather than First-Last.lower()
        if len(key.split()) > 1:
            if lower:
                val = join_by.join([k.lower() for k in key.split()])
            else:
                val = join_by.join(key.split())
            
            mapping_dict[key] = val

    # process the numpy array of records
    @np.vectorize
    def joining_complex_words(phrase):
        # for those transformed words
        # this function will map them
        # to their transformed counterpart
        # otherwise, just return original word
        try:
            return mapping_dict[phrase]
        except KeyError:
            if lower:
                return phrase.lower()
            return phrase

    # apply similar processing to targets
    def process_target(target):
        processed_target = target
        print("Words To Be Transformed Are:")
        for key, val in mapping_dict.items():
            if key in processed_target:
                print(key, end=" ")
            processed_target = processed_target.replace(key, val)
            
        return processed_target


    processed_source = joining_complex_words(records_array)
    processed_source = " ".join(processed_source.flatten().tolist())


    target = " ".join(example["summary"])
    processed_target = process_target(target)

    return processed_source, processed_target, mapping_dict


def process_example(example, join_by="[SPACE]", lower=False):
    processed_values = process_values(example)
    entities, num_attr_diff = process_entities(example)
    joint_attrs = process_types(example, num_attr_diff)
    home_or_aways = process_home_or_away(example)

    values_flatten = processed_values.values.flatten()
    entities_flatten = np.concatenate(entities)
    joint_attrs_flatten = np.concatenate(joint_attrs)
    home_or_aways_flatten = np.concatenate(home_or_aways)

    processed_example = np.stack(
        [values_flatten, entities_flatten,
         joint_attrs_flatten, home_or_aways_flatten], axis=-1)

    (processed_source,
     processed_target,
     mapping_dict) = process_compound_words(
        example, processed_example,
        join_by=join_by, lower=lower)

    return (processed_example,
            processed_source,
            processed_target,
            mapping_dict)


def preprocess_box_score(data, verbose=True):
    processed_sources = []
    processed_targets = []
    processed_examples = []
    compound_words_dicts = []
    for example in data:

        if not verbose:
            with misc_utils.suppress_stdout():
                (processed_example,
                 processed_source,
                 processed_target,
                 mapping_dict) = process_example(example)
        else:
            (processed_example,
             processed_source,
             processed_target,
             mapping_dict) = process_example(example)

        processed_sources.append(processed_source)
        processed_targets.append(processed_target)
        processed_examples.append(processed_example)
        compound_words_dicts.append(mapping_dict)

    return (processed_sources,
            processed_targets,
            processed_examples,
            compound_words_dicts)


def reverse_outputs(summaries, join_by="[SPACE]"):
    reversed_summaries = []
    for summary in summaries:
        reversed_summary = summary.replace(join_by, " ")
        reversed_summaries.append(reversed_summary)
    return reversed_summaries
