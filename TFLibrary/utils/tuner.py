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

    TODOs:
    1) change the filenames to be more informative
    2) Write process outputs to a text file
    3) add support for "evaluation_fn"
    """

    def __init__(self,
                 logdir,
                 config_file,
                 execute_file,
                 gpus=None,
                 evaluation_fn=None,
                 print_command=False):
        """Create a Tuner

        Args:
            config_file:
                String
                Directory to the parameter json file
            execute_file:
                String
                Directory to the executable shell file
            gpus:
                List of Strings
                GPU-IDs to be run in parallel
            evaluation_fn:
                Callable(hparams_instance)
                The function to be called after executing
                the tunee to collect evaluation results
                in arbitrary structure
        """
        if gpus and not isinstance(gpus, (list, tuple)):
            raise TypeError("`gpus` must be list or tuple")

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

        self._gpus = gpus
        self._logdir = logdir
        self._hparams = hparams
        self._executable = executable
        self._evaluation_fn = evaluation_fn
        self._print_command = print_command

        self._tmp_files = []
        self._instance_histories = []

        # retoration will overwrite setup
        self._restore_or_setup()

    @property
    def num_parallel(self):
        if self._gpus is not None:
            return len(self._gpus)
        return None

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

    def _create_tmp_file(self):
        tmp_file = tempfile.NamedTemporaryFile(
            prefix="TUNER_", dir=self._logdir, delete=False)
        self._tmp_files.append(tmp_file.name)
        return tmp_file.name

    def _execute_single_exe_instance(self, executable_instance):
        tmp_file = self._create_tmp_file()
        _run_single_command(
            tmp_file, executable_instance,
            print_command=self._print_command)

    def _execute_multiple_exe_instances(self, executable_instances):
        tmp_file = self._create_tmp_file()
        _run_multiple_commands(
            tmp_file, executable_instances,
            gpu_ids=self._gpus,
            print_command=self._print_command)

    def _clean_tmp_files(self):
        for f in self._tmp_files:
            os.remove(f)

    def tune(self):
        # iterate until the queue is empty

        if self.num_parallel:
            pbar = trange(0, len(self._instance_queue), self.num_parallel)
        else:
            pbar = trange(len(self._instance_queue))

        for _ in pbar:
            # pbar.set_description(message)
            if self.num_parallel:
                executable_instances = []
                for _ in range(self.num_parallel):
                    hparams_instance = self._instance_queue.popleft()
                    executable_instance = (
                        self._create_exe_instance(hparams_instance))
                    executable_instances.append(executable_instance)

                self._execute_multiple_exe_instances(executable_instances)

            else:
                # fetch one hparams
                hparams_instance = self._instance_queue.popleft()
                executable_instance = (
                    self._create_exe_instance(hparams_instance))

                # run the model
                self._execute_single_exe_instance(executable_instance)

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


def _run_single_command(fname, command, print_command=False):
    """Launch the process in a separate screen"""
    with open(fname, "w") as f:
        f.write("\n".join(command))

    command = "bash %s  >%s.log 2>&1 " % (fname, fname)

    if print_command:
        print("EXECUTING: \t " + command)

    misc_utils.run_command(command)


def _run_multiple_commands(fname, commands, gpu_ids=None, print_command=False):
    """http://www.shakthimaan.com/posts/2014/11/27/gnu-parallel/news.html"""
    if not gpu_ids:
        raise ValueError("In Single GPU setting, use _run_single_command")

    if not isinstance(gpu_ids, (list, tuple)):
        raise TypeError("`gpu_ids` must be list of GPU IDs")

    # e.g. FileName-0
    AddGpuIdToFileName = lambda gpu_id: "-".join([fname, gpu_id])

    # e.g. FileName-0, FileName-2, FileName-3
    for command, gpu_id in zip(commands, gpu_ids):
        with open(AddGpuIdToFileName(gpu_id), "w") as f:
            f.write("\n".join(command))

    # https://stackoverflow.com/questions/22187834/gnu-parallel-output-each-job-to-a-different-file
    # quote out the redirect
    command = (  # add --dry-run after `parallel` to them commands
        "parallel CUDA_VISIBLE_DEVICES=\"{}\" bash %s  \'>\'%s.log 2>&1 ::: %s"
        % (AddGpuIdToFileName("{}"),
           AddGpuIdToFileName("{}"),
           " ".join([i for i in gpu_ids])))

    if print_command:
        print("EXECUTING: \t " + command)

    misc_utils.run_command(command)
