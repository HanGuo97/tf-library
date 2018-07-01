import os
import json
import shutil
import itertools
from glob import glob
from TFLibrary.utils.tuner import Tuner


MODEL_PY = """import os
import argparse
import tempfile

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparam_0", type=str)
    parser.add_argument("--hparam_1", type=str)
    parser.add_argument("--hparam_2", type=float)
    parser.add_argument("--hparam_3", type=float)
    parser.add_argument("--logdir", type=str, default=None)
    
    FLAGS, unparsed = parser.parse_known_args()
    
    
    if FLAGS.logdir:
        fname = "HPS0_|%s|_HPS1_|%s|_HPS2_|%s|_HPS3_|%s|" % (
            str(FLAGS.hparam_0), str(FLAGS.hparam_1),
            str(FLAGS.hparam_2), str(FLAGS.hparam_3))

        fname = os.path.join(FLAGS.logdir, fname)
        with open(fname, "w") as f:
            f.write("")


if __name__ == "__main__":
    main()
"""


EXECUTABLE_SH = """#!/bin/bash

CONSTANT=0.3
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

CONFIG_JSON = {
    "TUNE_hparams_0": ["HPS_0_A", "HPS_0_B", "HPS_0_C"],
    "TUNE_hparams_1": ["HPS_1_A", "HPS_1_B"],
    "TUNE_hparams_2": [1.0, 2.0, 3.0, 4.0]}



def _create_test_files(logdir):
    with open(os.path.join(logdir, "model.py"), "w") as f:
        f.write(MODEL_PY)
    with open(os.path.join(logdir, "executable.sh"), "w") as f:
        f.write(EXECUTABLE_SH)
    with open(os.path.join(logdir, "config.json"), "w") as f:
        json.dump(CONFIG_JSON, f)
        

def _test_equal(logdir):
    # the filenames encodes the HPS's
    fname_pattern = os.path.join(logdir,
        "HPS0_|*|_HPS1_|*|_HPS2_|*|_HPS3_|*|")
    results = glob(fname_pattern)
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
        gpus=gpus)
    
    tuner.tune()
    _test_equal(logdir)
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
