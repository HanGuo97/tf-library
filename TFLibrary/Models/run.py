from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import argparse
import tensorflow as tf
from tensorflow.python.ops import lookup_ops
from TFLibrary.Seq2Seq import encoders
from TFLibrary.Data.utils import iterator_utils
from TFLibrary.Models import classification_models


def build_data(train_file, val_file,
               train_batch_size, val_batch_size,
               train_graph, val_graph):
    # get the UNK ID = Vocab Size + 1
    # but since Python is 0-based, UNK-ID = Vocab Size
    # and thus the new vocab size = old vocab size + 1
    with open(train_file + ".source_vocab") as f:
        _src_vocab = [d.strip() for d in f.readlines()]
        unk_id = len(_src_vocab)
        token_vocab_size = len(_src_vocab) + 1
        print("token_vocab_size is %d, UNK is %d" % (token_vocab_size, unk_id))

    # label vocab size == num_classes
    with open(train_file + ".label_vocab") as f:
        _tgt_vocab = [d.strip() for d in f.readlines()]
        label_vocab_size = len(_tgt_vocab)
        print("label_vocab_size is %d" % label_vocab_size)


    # train dataset
    with train_graph.as_default():
        # vocabs are stricted to train vocabs
        src_vocab_file = train_file + ".source_vocab"
        tgt_vocab_file = train_file + ".label_vocab"
        src_vocab_table = lookup_ops.index_table_from_file(
            src_vocab_file, default_value=unk_id)
        # no UNKs in target labels
        tgt_vocab_table = lookup_ops.index_table_from_file(tgt_vocab_file)

        train_src_1 = tf.data.TextLineDataset(train_file + ".sequence_1")
        train_src_2 = tf.data.TextLineDataset(train_file + ".sequence_2")
        train_tgt = tf.data.TextLineDataset(train_file + ".labels")
        train_batch = iterator_utils.get_pairwise_classification_iterator(
            src_dataset_1=train_src_1,
            src_dataset_2=train_src_2,
            tgt_dataset=train_tgt,
            src_vocab_table=src_vocab_table,
            tgt_vocab_table=tgt_vocab_table,
            batch_size=train_batch_size,
            padding_id=unk_id,
            random_seed=11)

    # val dataset
    with val_graph.as_default():
        # since these are graph-specific, we build them twice
        # vocabs are stricted to train vocabs
        src_vocab_file = train_file + ".source_vocab"
        tgt_vocab_file = train_file + ".label_vocab"
        src_vocab_table = lookup_ops.index_table_from_file(
            src_vocab_file, default_value=unk_id)
        # no UNKs in target labels
        tgt_vocab_table = lookup_ops.index_table_from_file(tgt_vocab_file)

        val_src_1 = tf.data.TextLineDataset(val_file + ".sequence_1")
        val_src_2 = tf.data.TextLineDataset(val_file + ".sequence_2")
        val_tgt = tf.data.TextLineDataset(val_file + ".labels")
        val_batch = iterator_utils.get_pairwise_classification_iterator(
            src_dataset_1=val_src_1,
            src_dataset_2=val_src_2,
            tgt_dataset=val_tgt,
            src_vocab_table=src_vocab_table,
            tgt_vocab_table=tgt_vocab_table,
            batch_size=val_batch_size,
            padding_id=unk_id,
            random_seed=11)

    return train_batch, val_batch, token_vocab_size, label_vocab_size


def build_model(data_batch,
                label_vocab_size,
                token_vocab_size,
                graph,
                logdir,
                is_training):
    
    model = classification_models.PairwiseClassificationModel(
        encoder_cls=encoders.LstmEncoder,
        data=data_batch,
        num_classes=label_vocab_size,
        token_vocab_size=token_vocab_size,
        token_embedding_size=128,
        # optimization
        optimizer=tf.train.RMSPropOptimizer,
        learning_rate=0.001,
        gradient_clipping_norm=2.0,
        # misc
        graph=graph,
        logdir=logdir,
        # encoder-specific
        unit_type="lstm",
        num_units=128,
        dropout_rate=0.5,
        is_training=is_training)
    
    model.build()
    return model


def train(FLAGS):
    train_graph = tf.Graph()
    val_graph = tf.Graph()
    
    (train_batch,
     val_batch,
     token_vocab_size,
     label_vocab_size) = build_data(
        train_file=FLAGS.train_file,
        val_file=FLAGS.val_file,
        train_batch_size=FLAGS.train_batch_size,
        val_batch_size=FLAGS.val_batch_size,
        train_graph=train_graph,
        val_graph=val_graph)

    train_model = build_model(
        data_batch=train_batch,
        token_vocab_size=token_vocab_size,
        label_vocab_size=label_vocab_size,
        graph=train_graph,
        logdir=FLAGS.logdir,
        is_training=True)

    val_model = build_model(
        data_batch=val_batch,
        token_vocab_size=token_vocab_size,
        label_vocab_size=label_vocab_size,
        graph=val_graph,
        logdir=FLAGS.logdir,
        is_training=False)

    train_model.initialize_or_restore_session()
    train_model.initialize_data_iterator()

    for _ in range(FLAGS.max_steps):
        try:
            train_model.train()
        except tf.errors.OutOfRangeError:
            train_model.initialize_data_iterator()
            continue

        if train_model.global_step % FLAGS.steps_per_eval == 0:
            train_model.save_session()
            tf.logging.info("Running Evaluation")
            val_model.initialize_or_restore_session()
            val_model.initialize_data_iterator()
            val_model.evaluate()


def infer(FLAGS):
    # use val as infer
    train_graph = tf.Graph()
    infer_graph = tf.Graph()
    
    (_,
     infer_batch,
     token_vocab_size,
     label_vocab_size) = build_data(
        train_file=FLAGS.train_file,
        val_file=FLAGS.val_file,
        train_batch_size=FLAGS.train_batch_size,
        val_batch_size=FLAGS.val_batch_size,
        train_graph=train_graph,
        val_graph=infer_graph)

    infer_model = build_model(
        data_batch=infer_batch,
        token_vocab_size=token_vocab_size,
        label_vocab_size=label_vocab_size,
        graph=infer_graph,
        logdir=FLAGS.logdir,
        is_training=False)

    tf.logging.info("Running Evaluation")
    infer_model.initialize_or_restore_session(ckpt_file=FLAGS.infer_ckpt)
    infer_model.initialize_data_iterator()
    infer_model.evaluate()


def add_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file",
                        type=str, default=None)
    parser.add_argument("--val_file",
                        type=str, default=None)

    parser.add_argument("--train_batch_size",
                        type=int, default=256)
    parser.add_argument("--val_batch_size",
                        type=int, default=1000)

    parser.add_argument("--max_steps",
                        type=int, default=5000)
    parser.add_argument("--steps_per_eval",
                        type=int, default=100)
    parser.add_argument("--logdir",
                        type=str, default=None)
    parser.add_argument("--infer",
                        action="store_true", default=False)
    parser.add_argument("--infer_ckpt",
                        type=str, default=None)
    
    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS


def main(unused_argv):
    tf.set_random_seed(111)
    FLAGS = add_arguments()

    if FLAGS.infer:
        infer(FLAGS)
    else:
        train(FLAGS)


if __name__ == '__main__':
    tf.app.run()
