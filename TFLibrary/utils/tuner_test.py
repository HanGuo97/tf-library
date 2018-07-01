import os
import json
import shutil
import itertools
from glob import glob
from collections import OrderedDict
from TFLibrary.utils.tuner import Tuner
from TFLibrary.utils.misc_utils import read_text_file


MODEL_PY = """import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparam_0", type=str)
    parser.add_argument("--hparam_1", type=str)
    parser.add_argument("--hparam_2", type=float)
    parser.add_argument("--hparam_3", type=float)
    parser.add_argument("--logdir", type=str, default=None)
    
    FLAGS, unparsed = parser.parse_known_args()
    
    fname = "HPS0_|%s|_HPS1_|%s|_HPS2_|%s|_HPS3_|%s|" % (
        str(FLAGS.hparam_0), str(FLAGS.hparam_1),
        str(FLAGS.hparam_2), str(FLAGS.hparam_3))

    if FLAGS.logdir:
        print(fname)
        fname = os.path.join(FLAGS.logdir, fname)
        with open(fname, "w") as f:
            f.write("")


if __name__ == "__main__":
    main()
"""


EXECUTABLE_SH = """#!/bin/bash

CONSTANT=0.333
LOGDIR="./TunerTest"
##################### TUNER

mkdir -p $LOGDIR

python ./TunerTest/model.py \\
    --hparam_0 $TUNE_hparams_1 \\
    --hparam_1 $TUNE_hparams_0 \\
    --hparam_2 $CONSTANT \\
    --hparam_3 $TUNE_hparams_2
    
    
python ./TunerTest/model.py \\
    --hparam_0 $TUNE_hparams_0 \\
    --hparam_1 $TUNE_hparams_1 \\
    --hparam_2 $TUNE_hparams_2 \\
    --hparam_3 $CONSTANT \\
    --logdir  $LOGDIR
"""

# use OrderDict so that the relative order won't change
CONFIG_JSON = OrderedDict([
    ("TUNE_hparams_0", ["HPS_0_A", "HPS_0_B", "HPS_0_C"]),
    ("TUNE_hparams_1", ["HPS_1_A", "HPS_1_B"]),
    ("TUNE_hparams_2", [1.0, 2.0, 3.0, 4.0])])


def _create_test_files(logdir):
    with open(os.path.join(logdir, "model.py"), "w") as f:
        f.write(MODEL_PY)
    with open(os.path.join(logdir, "executable.sh"), "w") as f:
        f.write(EXECUTABLE_SH)
    with open(os.path.join(logdir, "config.json"), "w") as f:
        json.dump(CONFIG_JSON, f)


def _test_equal(results):
    """Two Situations:
    
    1. If the results is a list of filenames of the form

        ./TunerTest/HPS0_|HPS_0_C|_HPS1_|HPS_1_B|_HPS2_|3.0|_HPS3_|0.3|
    
        >>>>> ["./TunerTest/HPS0_", "HPS_0_C", ..., "_HPS3_", "0.3", ""]

    2. If the results is a list of print-outs

        HPS0_|HPS_0_A|_HPS1_|HPS_1_A|_HPS2_|2.0|_HPS3_|0.3|
    
        >>>>> ["HPS0_", "HPS_0_B", "_HPS3_", "0.3", ""]

    """

    hps = sorted([[d for d in f.split("|") if (
                   not d.startswith("_") and
                   not d.endswith("_") and
                   len(d) > 0)] for f in results])
    # removing the constants I added
    hps = [d[:-1] for d in hps]

    # the HPS expected to be executed
    expected_hps = sorted(itertools.product(*CONFIG_JSON.values()))
    # string-ize all numerical values
    expected_hps = [[str(d) for d in ehps] for ehps in expected_hps]

    if hps == expected_hps:
        print("PASSED")
    else:
        print("hps\t", hps)
        print("expected_hps\t", expected_hps)
        raise ValueError("FAILED")


def _test_outputs(logdir):
    # the filenames encodes the HPS's
    fname_pattern = os.path.join(logdir,
        "HPS0_|*|_HPS1_|*|_HPS2_|*|_HPS3_|*|")
    results = glob(fname_pattern)
    _test_equal(results)


def _test_redirection(logdir):
    # the print-outs encodes the HPS's
    fname_pattern = os.path.join(logdir, "*.log")
    logfiles = glob(fname_pattern)
    results = [read_text_file(f)[0] for f in logfiles]
    _test_equal(results)


def _test(gpus=None, clean_after_test=False):
    logdir = "./TunerTest/"
    if not os.path.isdir(logdir):
        print("Creating ", logdir)
        os.mkdir(logdir)

    # create tmp files for testing
    _create_test_files(logdir=logdir)

    # initializ the tuner
    tuner = Tuner(
        logdir=logdir,
        config_file=os.path.join(logdir, "config.json"),
        execute_file=os.path.join(logdir, "executable.sh"),
        gpus=gpus, print_command=True)

    tuner.tune()
    _test_outputs(logdir)
    _test_redirection(logdir)
    if clean_after_test:
        print("Removing ", logdir)
        shutil.rmtree(logdir)


def test():
    _test(None, True)
    _test("0".split(","), True)
    _test("1,2".split(","), True)
    _test("0,2,3".split(","), True)


if __name__ == "__main__":
    test()
