import math
import tensorflow as tf
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
