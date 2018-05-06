import os
import sys
import random
import pickle
import numpy as np
from copy import deepcopy
from collections import deque
from contextlib import contextmanager


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
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


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
