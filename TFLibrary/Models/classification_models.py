from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import collections
import tensorflow as tf

from TFLibrary.Seq2Seq import base_models
from TFLibrary.utils import tensorflow_utils as tf_utils

tf.logging.set_verbosity(tf.logging.INFO)


class PairwiseClassificationModel(object):
    def __init__(self,
                 encoder_cls,
                 data,
                 num_classes,
                 token_vocab_size,
                 token_embedding_size,
                 # evaluation
                 evaluation_fn,
                 # optimization
                 optimizer="SGD",
                 learning_rate=0.001,
                 gradient_clipping_norm=2.0,
                 # misc
                 graph=None,
                 logdir=None,
                 debug_mode=False,
                 # encoder args
                 **encoder_kargs):
        """
        Classification model that does the mapping of

            f: Seq_1 x Seq_2 --> Class

        Args:
            encoder_cls:
                TFLibrary.Seq2Seq.base_models.BaseEncoder
                The class to encoder individual sequences
            data:
                TFLibrary.Data.utils.iterator_utils.BatchedInput
                Namedtuple containing tf.data.Dataset elements
            num_classes:
                Integer
                number of possible classes
            token_vocab_size:
                Integer
                vocabulary size for sequences
            token_embedding_size:
                Integer
                embedding size of tokens
            evaluation_fn:
                Callable([Prediction], [Seq_1], [Seq_2], [Target]) --> Float
                a function that takes list of outputs and inputs, and return
                a float of model performances
            optimizer:
                String or tf.train.Optimizer
                optimizer to be used
            learning_rate:
                Float
                learning rate
            gradient_clipping_norm:
                Float
                gradient clipping norm
            graph:
                tf.Graph
                Graph object used to construct the model, defaults
                to creating a new graph
            logdir:
                String
                directory to save model checkpoints and summaries
            debug_mode:
                Boolean
                whether to print out some additional information

            **encoder_kargs
                additional arguments passed to `encoder_cls`

        """

        if not issubclass(encoder_cls, base_models.BaseEncoder):
            raise TypeError(
                "Expected `core_encoder_cls` to be "
                "a subclass of base_models.BaseEncoder")
        if not callable(evaluation_fn):
            raise TypeError("`evaluation_fn` should be a callable function")
        
        self._encoder_cls = encoder_cls
        self._data = data
        self._num_classes = num_classes
        self._token_vocab_size = token_vocab_size
        self._token_embedding_size = token_embedding_size

        self._evaluation_fn = evaluation_fn

        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._gradient_clipping_norm = gradient_clipping_norm
        
        self._graph = graph or tf.Graph()
        self._logdir = logdir
        self._save_path = os.path.join(logdir, "model")
        self._summary_dir = os.path.join(logdir, "summaries")

        self._debug = {}
        self._debug_mode = debug_mode

        self._encoder_kargs = encoder_kargs

    def build(self):
        with self._graph.as_default():
            embedded_tokens_1, embedded_tokens_2 = self._build_inputs(
                tokens_1=self._data.source_1,
                tokens_2=self._data.source_2)

            logits = self._build_encoder(
                encoder_inputs_1=embedded_tokens_1,
                encoder_inputs_2=embedded_tokens_2,
                sequence_lengths_1=self._data.source_1_sequence_length,
                sequence_lengths_2=self._data.source_2_sequence_length)
    

            (train_op,
             cross_entropy,
             predictions,
             global_step_tensor) = self._build_loss(
                logits=logits,
                labels=self._data.target)

            summary_ops = tf.summary.merge_all()
            saver = tf.train.Saver(max_to_keep=20)
            file_writer = tf.summary.FileWriter(self._summary_dir)

        self._sess = None
        self._saver = saver
        self._file_writer = file_writer
        self._summary_ops = summary_ops
        self._cross_entropy = cross_entropy
        self._train_op = train_op
        self._predictions = predictions

        # first one is scalar
        self.global_step = None
        # second one is tensor
        self._global_step_tensor = global_step_tensor
    
    def initialize_or_restore_session(self, ckpt_file=None, initialize=True):
        # Initialize or restore session
        # restore from lastest_checkpoint or specific file
        with self._graph.as_default():
            self._sess = tf.Session(
                graph=self._graph,
                config=tf_utils.get_config())

            if initialize:
                self._sess.run(tf.tables_initializer())
                self._sess.run(tf.global_variables_initializer())

            if self._logdir or ckpt_file:
                # restore from lastest_checkpoint or specific file if provided
                tf_utils.load_ckpt(saver=self._saver,
                                   sess=self._sess,
                                   ckpt_dir=self._logdir,
                                   ckpt_file=ckpt_file)
                return

    def initialize_data_iterator(self):
        self._sess.run(self._data.initializer)

    def train(self):
        _, loss, summary, global_step = self._sess.run(
            [self._train_op,
             self._cross_entropy,
             self._summary_ops,
             self._global_step_tensor])

        # update latest global step
        self.global_step = global_step
        # write summaries
        self._file_writer.add_summary(summary, global_step=global_step)

        tf.logging.info("STEP %d LOSS: %.3f" % (self.global_step, loss))
        return loss

    def sample(self, include_data=False):
        """
        Fetch model outputs

        Returns:
            predictions:
                list: [batch size]
                outputs of model predictions
            fetched_data:
                tuple: [source_1, source_2, target]
                eachi of which is a list: [batch_size]
                input data that makes the predictions
        """
        if include_data:
            (source_1,
             source_2,
             target,
             predictions,
             global_step) = self._sess.run(
                [self._data.source_1,
                 self._data.source_2,
                 self._data.target,
                 self._predictions,
                 self._global_step_tensor])

            predictions = predictions.tolist()
            fetched_data = (source_1.tolist(),
                            source_2.tolist(),
                            target.tolist())

        else:
            fetched_data = None
            predictions, global_step = self._sess.run(
                [self._predictions,
                 self._global_step_tensor])

            predictions = predictions.tolist()

        # update latest global step
        self.global_step = global_step

        return predictions, fetched_data

    def evaluate(self):
        """Sample from model predictions, and evaluate outputs"""
        # predictions from the model
        all_predictions = []
        # fetched data that led to the predictions
        # dictionary of {seq_1: [], seq_2: [], target: []}
        all_fetched_data = collections.defaultdict(list)
        try:
            while True:
                # sample predictions
                predictions, fetched_data = self.sample(True)
                # cache the data
                all_predictions += predictions
                all_fetched_data["seq_1"] += fetched_data[0]
                all_fetched_data["seq_2"] += fetched_data[1]
                all_fetched_data["target"] += fetched_data[2]
        
        except tf.errors.OutOfRangeError:
            # evaluate
            counts = len(all_predictions)
            scores = self._evaluation_fn(
                all_predictions,
                all_fetched_data["seq_1"],
                all_fetched_data["seq_2"],
                all_fetched_data["target"])
            
            
        self.write_summary("ValScores", scores)
        tf.logging.info(
            "Validation Reports: \n"
            "\t\tSTEP %d\n\t\tCOUNT: %d\n\t\tScores %.2f"
            % (self.global_step, counts, scores))
        
        return scores

    def save_session(self):
        return self._saver.save(self._sess,
            save_path=self._save_path,
            global_step=self.global_step)

    def write_summary(self, tag, value):
        tf_utils.add_summary(tag=tag, value=value,
            summary_writer=self._file_writer,
            global_step=self.global_step)

    def _build_inputs(self, tokens_1, tokens_2):
        tokens_embedding = tf.get_variable(
            name="token_embeddings",
            dtype=tf.float32,
            shape=[self._token_vocab_size,
                   self._token_embedding_size])

        embedded_tokens_1 = tf.nn.embedding_lookup(
            params=tokens_embedding, ids=tokens_1)

        embedded_tokens_2 = tf.nn.embedding_lookup(
            params=tokens_embedding, ids=tokens_2)

        # for debugging
        self._tokens_embedding = tokens_embedding
        return embedded_tokens_1, embedded_tokens_2
        
    def _build_encoder(self,
                       encoder_inputs_1,
                       encoder_inputs_2,
                       sequence_lengths_1,
                       sequence_lengths_2):
        """
        BiLSTM with max pooling from
        https://arxiv.org/pdf/1705.02364.pdf
        """
        
        all_scopes = []
        all_encoders = []
        all_processed_outputs = []
        for i, (enc_inp, seq_len) in enumerate(zip(
                [encoder_inputs_1, encoder_inputs_2],
                [sequence_lengths_1, sequence_lengths_2])):

            with tf.variable_scope("Encoder_%d" % i) as scope:
                tf.logging.info("Creating %s" % self._encoder_cls.__name__)
                encoder = self._encoder_cls(**self._encoder_kargs)
                encoder.build()

                # here we hard-coded the use of bidir-LSTM
                # outputs: [batch_size, length, num_units x 2]
                outputs, states = encoder.encode(
                    inputs=enc_inp, sequence_length=seq_len)

                # row-level max-pooling
                # processed_outputs: [batch_size, num_units x 2]
                processed_outputs = tf.reduce_max(outputs, axis=1)

                # append the outputs
                all_scopes.append(scope)
                all_encoders.append(encoder)
                all_processed_outputs.append(processed_outputs)
        
        if len(all_processed_outputs) != 2:
            tf.logging.fatal("all_processed_outputs.len != 2")

        # [u, v, |u - v|, u * v]
        u = all_processed_outputs[0]
        v = all_processed_outputs[1]
        u_mul_v = tf.multiply(u, v)
        u_min_v = tf.abs(tf.subtract(u, v))
        features = tf.concat([u, v, u_min_v, u_mul_v], axis=-1)
        tf.logging.info("Using [u, v, |u - v|, u * v] as features")

        # final linear layer
        logits = tf.layers.dense(
            inputs=features,
            units=self._num_classes,
            name="LogitsLayer")

        # store the encoder for debugging
        self._scopes = all_scopes
        self._encoders = all_encoders
        self._debug["logits"] = logits

        return logits

    def _build_loss(self, logits, labels):
        
        if self._debug_mode:
            labels = tf.Print(labels, [labels],
                              "Labels", summarize=10)
        
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits))

        # Add the optimizer.
        # global_step_tensor = tf.Variable(
        #     0, name='global_step', trainable=False)
        global_step_tensor = tf.train.get_or_create_global_step(
            graph=self._graph)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.contrib.layers.optimize_loss(
                loss=cross_entropy,
                global_step=global_step_tensor,
                learning_rate=self._learning_rate,
                optimizer=self._optimizer,
                # some gradient clipping stabilizes training in the beginning.
                clip_gradients=self._gradient_clipping_norm,
                summaries=["learning_rate", "loss"])

        # Compute current predictions.
        predictions = tf.argmax(logits, axis=1)
        
        if self._debug_mode:
            predictions = tf.Print(
                predictions, [predictions],
                "Predictions ", summarize=10)
        
        # for debugging
        self._debug["labels"] = labels

        return train_op, cross_entropy, predictions, global_step_tensor
