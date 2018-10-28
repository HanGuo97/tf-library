import copy
from collections import OrderedDict
from TFLibrary.utils import misc_utils


INDICATOR = "##################### TUNER"


def _create_experiment(executable, hparams_instance):

    # insert one line below INDICATOR
    insert_index = executable.index(INDICATOR) + 1
    insertable = _create_bash_variable(hparams_instance)

    # create an instance of executable
    experiment = copy.deepcopy(executable)
    experiment[insert_index: insert_index] = insertable
    return experiment


def _create_bash_variable(d):
    """Given a dictionary of key, val, return a list pf
    bash-like varaible strings such as:

        key_0=val_0
        key_1=val_1
        ...
    """
    variables = []
    for key, val in d.items():
        variables.append("=".join(
            [key, _to_string(val)]))
    return variables


def _to_string(X):
    """Wrap X with quotes if X is string, otherwise str(X)"""
    if isinstance(X, str):
        return "\"%s\"" % X
    return str(X)


def merge_hparams(hparams_1, hparams_2):
    """Merge Hparams

    Currently only parallelization at the lowest level is
    supported, thus `hparams_1`, hparams coming from upper
    level, will never container more than one instance.
    `hparams_2`, however, could either be a OrderedDict, or
    a list of OrderedDict for parallelization.
    """
    if not isinstance(hparams_1, OrderedDict):
        raise TypeError("`hparams_1` should be `OrderedDict`, "
                        "but found %s" % type(hparams_1))
    if isinstance(hparams_2, OrderedDict):
        return misc_utils.merge_ordered_dicts(hparams_1, hparams_2)
    
    if (isinstance(hparams_2, (list, tuple)) and
            isinstance(hparams_2[0], OrderedDict)):
        return [misc_utils.merge_ordered_dicts(hparams_1, hp2)
                for hp2 in hparams_2]
        
    raise TypeError("`hparams_2` should be `OrderedDict`, or list "
                    "of `OrderedDict` but found %s" % type(hparams_2))
