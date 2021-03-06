from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import tensorflow as tf
from TFLibrary.utils import tensorflow_utils as tf_utils

tf.logging.set_verbosity(tf.logging.INFO)


class BaseModel(object):
    def __init__(self, logdir, graph=None, saver_max_to_keep=20):
        """Base Model"""
        self._logdir = logdir
        # directories to save checkpoints
        self._save_path = os.path.join(logdir, "model")
        # directories to save best checkpoints
        self._best_save_path = os.path.join(logdir, "best_model")
        # directories to save summaries
        self._summary_dir = os.path.join(logdir, "summaries")
        # graph object
        self._graph = graph or tf.Graph()
        # maximum number of checkpoints to save
        self._saver_max_to_keep = saver_max_to_keep

    def _check_compatability(self):
        """Check for some variables for compatability"""
        raise NotImplementedError

    def build(self):
        # check vars to make sure compatible
        self._check_compatability()

        with self._graph.as_default():
            self._build()
            saver = tf.train.Saver(
                max_to_keep=self._saver_max_to_keep)
            best_saver = tf.train.Saver(
                max_to_keep=self._saver_max_to_keep)
            file_writer = tf.summary.FileWriter(
                self._summary_dir)

        self._sess = None
        self._saver = saver
        self._best_saver = best_saver
        self._file_writer = file_writer

    def _build(self):
        """building Models"""
        raise NotImplementedError

    @property
    def global_step(self):
        return self._get_global_step()

    def _get_global_step(self):
        raise NotImplementedError

    def initialize_or_restore_session(self,
                                      ckpt_file=None,
                                      initialize=True,
                                      var_filter_fn=None):
        """Initialize or restore session,
           restore from lastest_checkpoint or specific file."""

        if not initialize and self._sess is None:
            raise ValueError("`self._sess` is not initialized yet")

        with self._graph.as_default():
            if initialize:
                self._sess = tf.Session(
                    graph=self._graph,
                    config=tf_utils.get_config())
                self._sess.run(tf.tables_initializer())
                self._sess.run(tf.global_variables_initializer())

            if var_filter_fn and callable(var_filter_fn):
                vars_to_restore = [v for v in tf.global_variables()
                                   if var_filter_fn(v.name)]
                saver = tf.train.Saver(var_list=vars_to_restore)
            else:
                saver = self._saver


            if self._logdir or ckpt_file:
                # restore from lastest_checkpoint
                # or specific file if provided
                tf_utils.load_ckpt(saver=saver,
                                   sess=self._sess,
                                   ckpt_dir=self._logdir,
                                   ckpt_file=ckpt_file)

    def save_session(self):
        return self._saver.save(
            self._sess,
            save_path=self._save_path,
            global_step=self.global_step)

    def save_best_session(self):
        return self._best_saver.save(
            self._sess,
            save_path=self._best_save_path,
            global_step=self.global_step)

    def write_summary(self, tag, value):
        tf_utils.add_summary(
            tag=tag, value=value,
            summary_writer=self._file_writer,
            global_step=self.global_step)
