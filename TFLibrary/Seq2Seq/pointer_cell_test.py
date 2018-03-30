import tensorflow as tf
# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution()
from TFLibrary.Seq2Seq.pointer_cell import PointerWrapper
from TFLibrary.SPG import pg_decoder
from TFLibrary.utils import test_utils
from TFLibrary.utils import attention_utils
from TFLibrary.Seq2Seq.customized_attention_wrapper import *


# Set Up ########################################
class Scope(object):
    Decoder = test_utils.create_scope("decoder")
    Attention = test_utils.create_scope("attention")
    Pointer = test_utils.create_scope("pointer")
    

class HPS(object):
    batch_size = 32
    enc_seq_length = 20
    dec_seq_length = 10
    embedding_size = 256
    num_units = 256
    vocab_size = 100
    OOVs = 100


class FakeVocab(object):
    def size(self):
        return HPS.vocab_size


def setup_pointer_cell():
    # attention
    cell = tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.BasicLSTMCell(HPS.num_units) for _ in range(2)])
    memories = test_utils.random_tensor(
        [HPS.batch_size, HPS.enc_seq_length, HPS.num_units])
    attention_mechanism = CustomizedBahdanauAttention(
        num_units=HPS.num_units, memory=memories, scope=Scope.Attention)
    # pointer
    enc_batch_extended_vocab = test_utils.random_integers(
        low=0, high=(HPS.vocab_size + HPS.OOVs),
        shape=[HPS.batch_size, HPS.enc_seq_length])
    # misc
    decoder_inputs = test_utils.random_tensor(
        [HPS.batch_size, HPS.dec_seq_length, HPS.embedding_size])
    enc_padding_mask = test_utils.random_integers(
        low=0, high=2, dtype=tf.float32,
        shape=[HPS.batch_size, HPS.enc_seq_length])

    pointer_cell = PointerWrapper(
        cell=cell,
        
        # attention
        attention_mechanism=attention_mechanism,
        attention_layer_size=cell.output_size,
        alignment_history=True,
        # cell_input_fn=None,
        # output_attention=True,
        
        # pointer
        coverage=False,
        batch_size=HPS.batch_size,
        vocab_size=HPS.vocab_size,
        num_source_OOVs=HPS.OOVs,
        enc_batch_extended_vocab=enc_batch_extended_vocab,
        # probability_fn=None,
        
        # misc
        initial_cell_state=None,
        name="pointer_cell",
        attention_scope=Scope.Attention,
        pointer_scope=Scope.Pointer)

    return (pointer_cell,
            cell,
            memories,
            attention_mechanism,
            enc_batch_extended_vocab,
            decoder_inputs,
            enc_padding_mask)


def get_session():
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    return sess


def check_tensor_equality(X, Y):
    sess = get_session()
    boolean_array = sess.run(tf.equal(X, Y))

    if boolean_array.all():
        print("PASSED")
    else:
        raise AssertionError("FAILED")


def test_attention():
    """TEST ATTENTION
    
        To test attention, we build another
        RNNCell using tf.contrib.seq2seq.AttentionWrapper
        and test our implementation aginst it
    """
    (pointer_cell,
     cell,
     memories,
     attention_mechanism,
     enc_batch_extended_vocab,
     decoder_inputs,
     enc_padding_mask) = setup_pointer_cell()
    
    attention_cell = CustomizedAttentionWrapper(
        cell=cell,
        attention_mechanism=attention_mechanism,
        alignment_history=True,
        attention_layer=pointer_cell._attention_layer)
    


    (poiner_cell_outputs,
     poiner_cell_state) = tf.nn.dynamic_rnn(
        cell=pointer_cell, inputs=decoder_inputs, dtype=tf.float32)

    (attention_cell_outputs,
     attention_cell_state) = tf.nn.dynamic_rnn(
        cell=attention_cell, inputs=decoder_inputs, dtype=tf.float32)
    

    check_tensor_equality(poiner_cell_outputs, attention_cell_outputs)
    check_tensor_equality(poiner_cell_state.attention,
                          attention_cell_state.attention)
    check_tensor_equality(poiner_cell_state.alignments,
                          attention_cell_state.alignments)
    check_tensor_equality(poiner_cell_state.alignment_history.stack(),
                          attention_cell_state.alignment_history.stack())
    for l in range(2):
        check_tensor_equality(poiner_cell_state[l].c,
                              attention_cell_state[l].c)
        check_tensor_equality(poiner_cell_state[l].h,
                              attention_cell_state[l].h)
    

    return attention_cell, attention_cell_outputs, attention_cell_state































