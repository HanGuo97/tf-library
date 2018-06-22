from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import shlex
import random
import pickle
import subprocess
import numpy as np
from copy import deepcopy
from collections import deque
from contextlib import contextmanager



def read_text_file(fname):
    with open(fname) as f:
        texts = [d.strip() for d in f.readlines()]

    return texts


def run_command(command):
    """https://zaiste.net/realtime_output_from_shell_command_in_python/"""
    process = subprocess.Popen(command,
        stdout=subprocess.PIPE, shell=True)
    while True:
        line = process.stdout.readline().rstrip()
        if not line:
            break
        print(line)


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def unique_ordered_list(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def depreciation_warning(cls):
    raise Exception("%s is depreciated" % cls.__name__)


@contextmanager
def calculate_time(tag):
    start_time = time()
    yield
    print("%s: " % tag, time() - start_time)


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def save_object(obj, filename):
    with open(filename, 'wb') as handle:  # Overwrites any existing file.
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def maybe_delete_file(file_dir, check_exists=False):
    if os.path.exists(file_dir):
        os.remove(file_dir)
        print("File %s is deleted" % file_dir)
    
    elif check_exists:
        raise ValueError("File %s does not exist" % file_dir)


def assert_all_same(items, attr=None):
    if not isinstance(items, (list, tuple)):
        raise TypeError("items should be list or tuple")

    if attr is not None:
        if not all(getattr(x, attr) == getattr(items[0], attr) for x in items):
            raise ValueError("items of %s not consistent between items" % attr)
    else:
        if not all(x == items[0] for x in items):
            raise ValueError("items not consistent between items")


def align_on_references(source_references, source_outputs, target_references):
    
    # make sure they are essentially two different orderings of the same thing
    if source_references == target_references:
        raise ValueError("References All Ready Aligned")
    if set(source_references) != set(target_references):
        raise ValueError("Two References Must Be Essentially The Same Thing")
    
    # make sure we don't destroy original files
    _source_references = deepcopy(source_references)
    _source_outputs = deepcopy(source_outputs)

    aligned_references = []
    aligned_outputs = []

    for ref in target_references:
        found_idx = _source_references.index(ref)

        aligned_references.append(_source_references[found_idx])
        aligned_outputs.append(_source_outputs[found_idx])

        # to avoid duplicate terms
        _source_references.pop(found_idx)
        _source_outputs.pop(found_idx)
        
        
    if not all([target_references == aligned_references,
                source_references != aligned_references,
                set(source_references) == set(aligned_references),
                set(source_outputs) == set(aligned_outputs)]):
        raise ValueError("Alignment Failed")
    
    return aligned_outputs


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
