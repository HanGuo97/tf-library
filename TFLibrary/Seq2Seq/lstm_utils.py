"""
https://github.com/tensorflow/magenta/blob/master/magenta/models/music_vae/lstm_utils.py
"""

import tensorflow as tf


def maybe_split_sequence_lengths(sequence_length, num_splits, total_length):
  """Validates and splits `sequence_length`, if necessary.
  Returned value must be used in graph for all validations to be executed.
  Args:
    sequence_length: A batch of sequence lengths, either sized `[batch_size]`
      and equal to either 0 or `total_length`, or sized
      `[batch_size, num_splits]`.
    num_splits: The scalar number of splits of the full sequences.
    total_length: The scalar total sequence length (potentially padded).
  Returns:
    sequence_length: If input shape was `[batch_size, num_splits]`, returns the
      same Tensor. Otherwise, returns a Tensor of that shape with each input
      length in the batch divided by `num_splits`.
  Raises:
    ValueError: If `sequence_length` is not shaped `[batch_size]` or
      `[batch_size, num_splits]`.
    tf.errors.InvalidArgumentError: If `sequence_length` is shaped
      `[batch_size]` and all values are not either 0 or `total_length`.
  """
  if sequence_length.shape.ndims == 1:
    if total_length % num_splits != 0:
      raise ValueError(
          '`total_length` must be evenly divisible by `num_splits`.')
    with tf.control_dependencies(
        [tf.Assert(
            tf.reduce_all(
                tf.logical_or(tf.equal(sequence_length, 0),
                              tf.equal(sequence_length, total_length))),
            data=[sequence_length])]):
      sequence_length = (
          tf.tile(tf.expand_dims(sequence_length, axis=1), [1, num_splits]) //
          num_splits)
  elif sequence_length.shape.ndims == 2:
    with tf.control_dependencies([
        tf.assert_less_equal(
            sequence_length,
            tf.constant(total_length // num_splits, tf.int32),
            message='Segment length cannot be more than '
                    '`total_length / num_splits`.')]):
      sequence_length = tf.identity(sequence_length)
    sequence_length.set_shape([sequence_length.shape[0], num_splits])
  else:
    raise ValueError(
        'Sequence lengths must be given as a vector or a 2D Tensor whose '
        'second dimension size matches its initial hierarchical split. Got '
        'shape: %s' % sequence_length.shape.as_list())
  return sequence_length