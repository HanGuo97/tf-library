from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import re
import os
import copy
import json
import pickle
import tempfile
import warnings
import itertools
import subprocess
from tqdm import trange
from collections import OrderedDict, deque
from TFLibrary.utils import misc_utils


INDICATOR = "##################### TUNER"


class Tuner(object):
    """Hyper-parameter Tuner.

    The Tuner receives as inputs of two files:
        1. config.json:
            A JSON file specifying the hyperparameter search
            space. It should come with the format:
            {ParameterA:[value_0, value_1, value_2 , ...]
             ParameterB:[value_0, value_1, value_2 , ...]}

        2. execute.sh:
            A bash script that executes the training process.
            It should come with the format:

            1. `# TUNER` to indicate where variables will be asigned
            2. As of now, only python script is supported, and each line
                can only assign one varaibles
            
            # TUNER

            python model.py \
                --hparam_0 $TUNE_hparams_0 \
                --hparam_1 $TUNE_hparams_1 \
                --hparam_2 $TUNE_hparams_2 \
                ...

            where variables starting with "TUNE_" will be replaced
            with actual values as specified in config.json

    TODOs: handle crash during tuning
    """

    def __init__(self,
                 logdir,
                 config_file,
                 execute_file,
                 evaluation_fn=None):

        """Create a Tuner

        Args:
            config_file:
                String
                Directory to the parameter json file
            execute_file:
                String
                Directory to the executable shell file
            evaluation_fn:
                Callable(hparams_instance)
                The function to be called after executing
                the tunee to collect evaluation results
                in arbitrary structure
        """
        if evaluation_fn and not callable(evaluation_fn):
            raise TypeError("`evaluation_fn` must be callable")

        # read the json file into OrderedDict
        # using OrderedDict to prevent accident re-ordering
        # of key-val pairs in later stages
        with open(config_file) as f:
            hparams = OrderedDict(**json.load(f))

        with open(execute_file) as f:
            executable = [d.strip("\n") for d in f.readlines()]

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        self._logdir = logdir
        self._hparams = hparams
        self._executable = executable
        self._evaluation_fn = evaluation_fn
        
        self._tmp_files = []
        self._instance_histories = []

        # retoratuin will overwrite setup
        self._restore_or_setup()
    
    def _restore_or_setup(self):
        try:
            self.restore_tuner()
            warnings.warn("Restoring the TUNER from %s" % self._logdir)
        except FileNotFoundError:
            self._setup()


    def _setup(self):
        """Setting up the tuner"""
        instances = _generate_hparam_instances(self._hparams)
        instance_queue = deque(instances)
        self._instances = instances
        self._instance_queue = instance_queue

    def _create_exe_instance(self, hparams_instance):
        # insert one line below INDICATOR
        insert_index = self._executable.index(INDICATOR) + 1
        insertable = _create_bash_variable(hparams_instance)
        
        # create an instance of executable
        executable_instance = copy.deepcopy(self._executable)
        executable_instance[insert_index: insert_index] = insertable
        return executable_instance

    def _execute_exe_instance(self, executable_instance):
        tmp_file = tempfile.NamedTemporaryFile(
            prefix="TUNER_", suffix=".sh",
            dir=self._logdir, delete=False)

        self._tmp_files.append(tmp_file.name)
        _run_command(tmp_file.name, executable_instance)

    def clean_tmp_files(self):
        for f in self._tmp_files:
            os.remove(f)


    def tune(self):
        # iterate until the queue is empty

        pbar = trange(len(self._instance_queue))
        for _ in pbar:
            # pbar.set_description(message)
            
            # fetch one hparams
            hparams_instance = self._instance_queue.popleft()
            executable_instance = self._create_exe_instance(hparams_instance)

            # run the model
            self._execute_exe_instance(executable_instance)

            # collect evaluation results
            if self._evaluation_fn is not None:
                evaluation_results = self._evaluation_fn(
                    hparams_instance)
                # cache the history
                self._instance_histories.append(
                    [hparams_instance, evaluation_results])

            self.save_tuner()


    def save_tuner(self):
        queue_fname = os.path.join(self._logdir, "TUNER.queue")
        history_fname = os.path.join(self._logdir, "TUNER.history")
        misc_utils.save_object(self._instance_queue, queue_fname)
        misc_utils.save_object(self._instance_histories, history_fname)


    def restore_tuner(self):
        queue_fname = os.path.join(self._logdir, "TUNER.queue")
        history_fname = os.path.join(self._logdir, "TUNER.history")
        self._instance_queue = misc_utils.load_object(queue_fname)
        self._instance_histories = misc_utils.load_object(history_fname)

        
def _generate_hparam_instances(d):
    # generate all combinations of dictionary values, unnamed
    value_collections = itertools.product(*d.values())
    # map the combination of values into a named dictionary
    # using OrderedDict simply to be consistent, but dict() also works
    hparam_collections = [OrderedDict((k, v)
                          for k, v in zip(d.keys(), vals))
                          for vals in value_collections]
    return hparam_collections


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


def _run_command(fname, command):
    """Launch the process in a separate screen"""
    with open(fname, "w") as f:
        f.write("\n".join(command))
    # print("EXECUTING: \t " + fname)
    misc_utils.run_command("bash " + fname)
