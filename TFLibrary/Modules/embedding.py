import math
from warnings import warn
import tensorflow as tf
import tensorflow_hub as tf_hub
from tensorflow.python.ops import lookup_ops
from TFLibrary.Modules import base


def _embedding_dim(vocab_size):
    """Calculate a reasonable embedding size for a vocabulary.
    Rule of thumb is 6 * 4th root of vocab_size.
    Args:
      vocab_size: Size of the input vocabulary.
    Returns:
      The embedding size to use.
    Raises:
      ValueError: if `vocab_size` is invalid.
    """
    if not vocab_size or (vocab_size <= 0):
        raise ValueError("Invalid vocab_size %g." % vocab_size)
    return int(round(6.0 * math.sqrt(math.sqrt(vocab_size))))


class Embeddding(base.AbstractModule):
    """Module for embedding tokens in a low-dimensional space."""

    def __init__(self,
                 vocab_size,
                 embed_dim=None,
                 trainable=True,
                 name="embed"):
        super(Embeddding, self).__init__(name=name)
        self._vocab_size = vocab_size
        self._embed_dim = embed_dim or _embedding_dim(self._vocab_size)
        self._trainable = trainable

    def _build(self, ids):
        """Lookup embeddings."""
        self._embeddings = tf.get_variable(
            "embeddings",
            shape=[self._vocab_size, self._embed_dim],
            dtype=tf.float32, trainable=self._trainable)

        # Lookup embeddings
        return tf.nn.embedding_lookup(
            self._embeddings, ids, name="embedding_lookup")

    @property
    def vocab_size(self):
        """Size of input vocabulary."""
        return self._vocab_size

    @property
    def embed_dim(self):
        """Size of embedding vectors."""
        return self._embed_dim

    @property
    def embeddings(self):
        """Returns the Variable containing embeddings."""
        return self._embeddings

    def _clone(self, name):
        return type(self)(vocab_size=self._vocab_size,
                          embed_dim=self._embed_dim,
                          trainable=self._trainable,
                          name=name)


class TFHubElmoEmbedding(base.AbstractModule):
    """Module for embdding tokens using TF-Hub ELMO"""
    ELMO_URL = "https://tfhub.dev/google/elmo/2"

    def __init__(self,
                 vocab_file,
                 trainable=False,
                 name="elmo_embed"):
        super(TFHubElmoEmbedding, self).__init__(name=name)
        warn("Not Unit Test Has Been Done To Ensure Correctness")
        
        hub = tf_hub.Module(self.ELMO_URL, trainable=trainable)
        self._elmo = hub
        self._reverse_vocab = (
            lookup_ops.index_to_string_table_from_file(vocab_file))

    def _build(self, tokens_input, tokens_length):
        processed_input = tf.to_int64(tokens_input)
        processed_input = self._reverse_vocab.lookup(processed_input)
        embeddings = self._elmo(
            inputs={"tokens": processed_input,
                    "sequence_len": tokens_length},
            signature="tokens",
            as_dict=True)["elmo"]

        return embeddings

    def _clone(self, name):
        raise NotImplementedError
