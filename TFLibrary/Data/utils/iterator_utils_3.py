"""
Need to make sure Target indices match Source indices

"""


import collections
import tensorflow as tf


# NOTE(ebrevdo): When we subclass this, instances' __dict__ becomes empty.
class BatchedInput3(
    collections.namedtuple("BatchedInput3",
        ("initializer",
         "source_1", "source_2", "target",
         "source_1_sequence_length",
         "source_2_sequence_length",
         "target_sequence_length"))):
    pass


def get_pairwise_classification_iterator(
        src_dataset_1,
        src_dataset_2,
        tgt_dataset,
        tgt_vocab_table,
        batch_size,
        random_seed,
        src_max_len=None,
        num_parallel_calls=4,
        output_buffer_size=None,
        skip_count=None,
        num_shards=1,
        shard_index=0,
        shuffle=True,
        repeat=False,
        reshuffle_each_iteration=True):
    
    if (src_dataset_1.output_shapes.ndims != 2 and
            src_dataset_2.output_shapes.ndims != 2):
        raise ValueError(
            "`src_dataset_1` and `src_dataset_2` must have ndims = 2 "
            "found %d and %d" % (src_dataset_1.output_shapes.ndims,
                                 src_dataset_2.output_shapes.ndims))

    if not output_buffer_size:
        output_buffer_size = batch_size * 1000

    src_tgt_dataset = tf.data.Dataset.zip(
        (src_dataset_1, src_dataset_2, tgt_dataset))

    src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)
    if skip_count is not None:
        src_tgt_dataset = src_tgt_dataset.skip(skip_count)

    if shuffle:
        src_tgt_dataset = src_tgt_dataset.shuffle(
            output_buffer_size, random_seed, reshuffle_each_iteration)

    # Filter zero length input sequences.
    src_tgt_dataset = src_tgt_dataset.filter(
        lambda src_1, src_2, tgt: (
            tf.logical_and(
                tf.logical_and(tf.size(src_1) > 0,
                               tf.size(src_2) > 0),
                tf.size(tgt) > 0)))

    # not target max len because this is classification
    if src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src_1, src_2, tgt: (
                src_1[:src_max_len], src_2[:src_max_len], tgt),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)


    # Convert the word strings to ids. Word strings that are not in the
    # vocab get the lookup table's default_value integer.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src_1, src_2, tgt: (
            src_1, src_2,
            tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    
    # Add in sequence lengths.
    # target lengths are redundant, but kept
    # for consistency
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src_1, src_2, tgt: (
            src_1, src_2, tgt,
            # [sequence_length, num_units]
            tf.shape(src_1)[0], tf.shape(src_2)[0], tf.size(tgt)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    if repeat:
        # Repeat the input indefinitely.
        src_tgt_dataset = src_tgt_dataset.repeat()

    # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The first three entries are the source and target line rows;
            # these have unknown-length vectors.  The last two entries are
            # the source and target row sizes; these are scalars.
            padded_shapes=(
                tf.TensorShape([None, None]),  # src_1
                tf.TensorShape([None, None]),  # src_2
                tf.TensorShape([]),  # tgt
                tf.TensorShape([]),  # src_1_len
                tf.TensorShape([]),  # src_2_len
                tf.TensorShape([])),  # tgt_len
            # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true
            # sequence.
            padding_values=(
                0.0,  # src_1
                0.0,  # src_2
                0,  # tgt -- unused
                0,  # src_1_len -- unused
                0,  # src_2_len -- unused
                0))  # tgt_len -- unused
    
    batched_dataset = batching_func(src_tgt_dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    (src_1_ids, src_2_ids, tgt_ids,
     src_1_seq_len, src_2_seq_len, tgt_seq_len) = (batched_iter.get_next())
    
    return BatchedInput3(
        initializer=batched_iter.initializer,
        source_1=src_1_ids,
        source_2=src_2_ids,
        target=tgt_ids,
        source_1_sequence_length=src_1_seq_len,
        source_2_sequence_length=src_2_seq_len,
        target_sequence_length=tgt_seq_len)
