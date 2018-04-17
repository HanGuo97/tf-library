from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import collections
from TFLibrary.utils import hparams_utils
from tensorflow.contrib.training import HParams

TrainingLog = collections.namedtuple("TrainingLog",
    ("BestValue", "ValueHistory", "BestCheckpoint"))


def _early_stop_with_tolerance(tolerance=5):
    def _stopping_fn(best, history):
        # initially None
        if not best:
            return False
        # initially list()
        if not history:
            return False
        # too short
        if len(history) < tolerance:
            return False

        return best not in history[-tolerance:]

    print("Early Stop with Tolerance %d" % tolerance)
    return _stopping_fn


def _greedy_update():
    def _updating_fn(best, history, value):
        # initially None
        if not best:
            return True
        # initially list()
        if not history:
            return True
        return value > best

    return _updating_fn


class TrainingManager(object):
    """
    Training Manager for Multiple Models

    It logs the dictionary {
        Name: TrainingLog(
            BestValue: best value throughout the training,
            ValueHistory: list of history of values,
            BestCheckpoint: checkpoint corresponding to best value)
    }
    """

    def __init__(self, name, logdir,
                 stopping_fn=_early_stop_with_tolerance(5),
                 updating_fn=_greedy_update(),
                 load_when_possible=True):
        """
        Initialize the TrainingManager

        Args:
            name:
                String
                name of the model
            logdir:
                Directory to the folder where manager will be saved
            stopping_fn:
                Callable (best_value, value_history): Boolean
                The function that determines when training
                should stop
                Common choices include:
                    lambda best, history: best not in history[-N:]
            updating_fn:
                Callable (best_value, value_history, new_value): Boolean
                The function that determines when the best
                value will be updated.
                Common choices include:
                    lambda best, history, value: value > best
        """
        if not callable(stopping_fn):
            raise TypeError("`stopping_fn` should be callable")
        if not callable(updating_fn):
            raise TypeError("`updating_fn` should be callable")
        
        # create log file
        logfile = os.path.join(logdir, "training_manager")

        # initialize the training logs
        training_logs = HParams()
        training_logs.add_hparam(
            name=name,
            value=TrainingLog(
                BestValue=None,
                ValueHistory=list(),
                BestCheckpoint=None))

        self._name = name
        self._logdir = logdir
        self._logfile = logfile
        self._stopping_fn = stopping_fn
        self._updating_fn = updating_fn
        self._training_logs = training_logs

        if load_when_possible:
            self.load()

    def get_training_log(self, key, default=None):
        return self._training_logs.get(key, default=None)

    def set_training_log(self, key, value):
        setattr(self._training_logs, key, value)

    @property
    def best_checkpoint(self):
        training_log = self.get_training_log(self._name)
        return training_log.BestCheckpoint

    @property
    def should_stop(self):
        training_log = self.get_training_log(self._name)
        best_value = training_log.BestValue
        history = training_log.ValueHistory
        return self._stopping_fn(best_value, history)

    def update(self, value, ckpt, verbose=False):
        training_log = self.get_training_log(self._name)
        best_value = training_log.BestValue
        history = training_log.ValueHistory
        best_checkpoint = training_log.BestCheckpoint

        new_history = training_log.ValueHistory + [value]
        if self._updating_fn(best_value, history, value):
            new_value = value
            new_ckpt = ckpt
        else:
            new_value = best_value
            new_ckpt = best_checkpoint
        
        new_training_log = TrainingLog(
            BestValue=new_value,
            ValueHistory=new_history,
            BestCheckpoint=new_ckpt)

        self.set_training_log(
            key=self._name,
            value=new_training_log)

        if verbose:
            self.print_info()

    def print_info(self):
        training_log = self.get_training_log(self._name)
        print("TrainingManager INFO:\n",
              "BestValue: %.2f\n" % training_log.BestValue,
              "ValueHistory: %s\n" % training_log.ValueHistory[:-3],
              "BestCheckpoint: %.2f\n" % training_log.BestCheckpoint)

    def save(self):
        hparams_utils.save_hparams(
            hparams_file=self._logfile,
            hparams=self._training_logs)

    def load(self):
        # load if file exists, else None
        training_logs = hparams_utils.load_hparams(self._logfile)
        if training_logs is not None:
            # when loaded, namedtuple will be automatically
            # cast into lists , so we need to convert them back
            values = training_logs.values()
            for key in sorted(values.keys()):
                training_log = TrainingLog(* values[key])
                self.set_training_log(key=key, value=training_log)
