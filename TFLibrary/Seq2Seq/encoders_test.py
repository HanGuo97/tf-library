import numpy as np
import tensorflow as tf
from TFLibrary.utils import test_utils
from TFLibrary.Seq2Seq.encoders import LstmEncoder
from TFLibrary.Seq2Seq.encoders import HierarchicalLstmEncoder

# tf.enable_eager_execution()


def test():
    test_HierarchicalLstmEncoder(True)
    test_HierarchicalLstmEncoder(False)


def test_HierarchicalLstmEncoder(full_level=True):
    num_levels = 4 if full_level else 3

    num_units = 16
    batch_size = 4
    sequence_length = 100
    sequence_lengths = np.tile(sequence_length, batch_size)
    sequence_lengths = tf.convert_to_tensor(sequence_lengths)
    sequence = test_utils.random_tensor(
        [batch_size, sequence_length, num_units])

    # Actual Outputs
    # ------------------------------------------
    # level 0: 20 chunks of 5-length sequences
    # level 1: 4 chunks of 5-length sequences
    # level 2: 2 chunks of 2-length sequences
    # outputs: 2-length sequence
    encoder = HierarchicalLstmEncoder(
        core_encoder_cls=LstmEncoder,
        level_lengths=[5, 5, 2, 2][:num_levels],
        total_length=sequence_length,
        unit_type="lstm",
        num_units=16)
    encoder.build()
    outputs, states = encoder.encode(
        sequence=sequence,
        sequence_length=sequence_lengths)

    # Expected Outputs
    # ------------------------------------------
    _sequence = sequence
    lens = [5, 5, 2, 2]
    num_splits = [20, 4, 2, 1]
    for i in range(num_levels):
        level_i_lens = lens[i]
        level_i_num_splits = num_splits[i]

        seq_lens = tf.fill(dims=[batch_size], value=level_i_lens)
        level_i_sequences = tf.split(
            _sequence, num_or_size_splits=level_i_num_splits, axis=1)
        level_i_output_tuples = [encoder.level(i).encode(seq, seq_lens)
                                 for seq in level_i_sequences]
        level_i_outputs = [o[0] for o in level_i_output_tuples]
        level_i_final_states = [o[1] for o in level_i_output_tuples]
        level_i_hidden = [tf.concat([s.h for s in stup], axis=-1)
                          for stup in level_i_final_states]
        
        # pass last hidden states to next level
        _sequence = tf.stack(level_i_hidden, axis=1)

    # Check Differences
    # ------------------------------------------
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    actual_outputs_tensor = tf.stack(outputs, axis=-1)
    actual_states_tensor = tf.stack(states, axis=-1)
    expected_outputs_tensor = tf.stack(level_i_outputs, axis=-1)
    expected_states_tensor = tf.stack(level_i_final_states, axis=-1)

    (actual_outputs, actual_states,
     expected_outputs, expected_states) = sess.run(
        [actual_outputs_tensor, actual_states_tensor,
         expected_outputs_tensor, expected_states_tensor])


    test_utils.check_equality(actual_outputs, expected_outputs)
    test_utils.check_equality(actual_states, expected_states)
    print("len(outputs) = %d %d \tlen(states) %d %d" %
        (len(outputs), len(level_i_outputs),
         len(states), len(level_i_final_states)))
