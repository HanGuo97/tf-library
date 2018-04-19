from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import collections
from TFLibrary.utils import hparams_utils
from tensorflow.contrib.training import HParams

ALLOWED_VALUE_TYPES = (dict)


class TrainingManager(object):
    """
    Training Manager for Multiple Models

    It logs the dictionary of the structure:
    
    Train:
        Name_BestValue: dictionary of numbers
        Name_ValueHistory: dictionary of list of floats
        Name_BestCheckpoint: checkpoint corresponding to Name_BestValue
    
    Test:
        Name_Value: float or a dictionary of floats

    And saves the files into /logdir/train.log, /logdir/test.log
    """

    def __init__(self, name, logdir,
                 stopping_fn, updating_fn,
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
        # never stop and always update
        if stopping_fn is None:
            stopping_fn = lambda best, history: False
        if updating_fn is None:
            updating_fn = lambda best, history, value: True

        # check types
        if not callable(stopping_fn):
            raise TypeError("`stopping_fn` should be callable")
        if not callable(updating_fn):
            raise TypeError("`updating_fn` should be callable")

        # initialize the logs
        test_logs = HParams()
        train_logs = HParams()
        train_logs.add_hparam("_".join([name, "BestValue"]), dict())
        train_logs.add_hparam("_".join([name, "ValueHistory"]),
                              collections.defaultdict(list))
        train_logs.add_hparam("_".join([name, "BestCheckpoint"]), None)
        
        # create log file
        test_logfile = os.path.join(logdir, "test.log")
        train_logfile = os.path.join(logdir, "train.log")
        

        self._name = name
        self._stopping_fn = stopping_fn
        self._updating_fn = updating_fn

        self._logdir = logdir
        self._test_logfile = test_logfile
        self._train_logfile = train_logfile
        
        self._test_logs = test_logs
        self._train_logs = train_logs
        

        if load_when_possible:
            self.load()

    def get_train_attr(self, attr, default=None):
        attr_name = "_".join([self._name, attr])
        return self._train_logs.get(attr_name, default)

    def set_train_attr(self, attr, value):
        attr_name = "_".join([self._name, attr])
        self._train_logs.set_hparam(attr_name, value)

    @property
    def best_value(self):
        return self.get_train_attr("BestValue")

    def set_best_value(self, value):
        self.set_train_attr("BestValue", value)

    @property
    def value_history(self):
        return self.get_train_attr("ValueHistory")

    def append_to_history(self, value):
        # since value_history is a defaultdict
        # we just append directly without checking
        for k, v in value.items():
            self.value_history[k].append(v)

    @property
    def best_checkpoint(self):
        return self.get_train_attr("BestCheckpoint")

    def set_best_checkpoint(self, ckpt):
        self.set_train_attr("BestCheckpoint", ckpt)


    @property
    def should_stop(self):
        # whether these dicts are empty
        if not self.best_value:
            return False
        if not self.value_history:
            return False

        return self._stopping_fn(self.best_value,
                                 self.value_history)

    def should_update(self, value):
        # whether these dicts are empty
        if not self.best_value:
            return True
        if not self.value_history:
            return True
        return self._updating_fn(self.best_value,
                                 self.value_history, value)

    def update(self, value, ckpt, verbose=False):
        """
        Update the manager. `value` must be of the same type
        of `train_logs.best_value`. For example, if `train_logs.best_value`
        is float, then `value` must be float. If `train_logs.best_value`
        is a dictionary, then `value` must also be a dictionary of
        with same keys
        """
        if not isinstance(value, ALLOWED_VALUE_TYPES):
            raise TypeError("`value` must be allowed, "
                            "found ", type(value))

        if self.should_update(value):
            self.set_best_value(value)
            self.set_best_checkpoint(ckpt)

        self.append_to_history(value)

        if verbose:
            self.print_info()

    def print_info(self):
        value_str = "".join(["%s %s " % (key, val)
            for key, val in self.best_value.items()])

        history_str = ""
        for key, val in self.value_history.items():
            vstr = " ".join((str(round(v, 3)) for v in val[-3:]))
            history_str += "%s %s " % (key, vstr)

        print("TrainingManager INFO:\n",
              "BestValue: %s\n" % value_str,
              "ValueHistory: %s\n" % history_str,
              "BestCheckpoint: %s\n" % self.best_checkpoint)

    def save(self):
        hparams_utils.save_hparams(
            hparams_file=self._train_logfile,
            hparams=self._train_logs)

    def load(self, check_exists=False):
        # load if file exists, else None
        loaded_logs = hparams_utils.load_hparams(self._train_logfile)
        if check_exists and loaded_logs is None:
            raise IOError("`logs` from %s cannot "
                "be loaded" % self._train_logfile)
        
        if loaded_logs is None:
            return

        for key, val in loaded_logs.values().items():
            try:  # create a new hps
                self._train_logs.add_hparam(key, val)
                print("ADDED HPS: %s" % key)
            except ValueError:  # if this hps exists
                # using set_hparam caused some bugs
                # so I am forcefully using setattr
                setattr(self._train_logs, key, val)
                print("LOADED HPS: %s" % key)
