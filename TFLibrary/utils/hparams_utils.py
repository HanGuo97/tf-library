from __future__ import print_function

import codecs
import collections
import json
import math
import os
import sys
import time

import numpy as np
import tensorflow as tf


def print_hparams(hparams, skip_patterns=None, header=None):
    """Print hparams, can skip keys based on pattern."""
    if header:
        print("%s" % header)
    values = hparams.values()
    for key in sorted(values.keys()):
        if not skip_patterns or all(
                [skip_pattern not in key for skip_pattern in skip_patterns]):
            print("  %s=%s" % (key, str(values[key])))


def load_hparams(hparams_file):
    """Load hparams from an existing model directory."""
    if tf.gfile.Exists(hparams_file):
        print("# Loading hparams from %s" % hparams_file)
        with codecs.getreader("utf-8")(tf.gfile.GFile(hparams_file, "rb")) as f:
            try:
                hparams_values = json.load(f)
                hparams = tf.contrib.training.HParams(**hparams_values)
            except ValueError:
                print("  can't load hparams file")
                return None
        return hparams
    else:
        return None


def save_hparams(hparams_file, hparams):
    """Save hparams."""
    print("  saving hparams to %s" % hparams_file)
    with codecs.getwriter("utf-8")(tf.gfile.GFile(hparams_file, "wb")) as f:
        f.write(hparams.to_json())


def maybe_parse_standard_hparams(hparams, hparams_path):
    """Override hparams values with existing standard hparams config."""
    if not hparams_path:
        return hparams

    if tf.gfile.Exists(hparams_path):
        print("# Loading standard hparams from %s" % hparams_path)
        with tf.gfile.GFile(hparams_path, "r") as f:
            hparams.parse_json(f.read())

    return hparams
