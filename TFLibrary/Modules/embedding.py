import math
import tensorflow as tf
import tensorflow_hub as tf_hub
from TFLibrary.Modules import base
from TFLibrary.Modules import utils


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
    """Module for embedding tokens in a low-dimensional space.

        TOPO:
        Add Initializer to the embedding
    """

    def __init__(self,
                 vocab_size=None,
                 embed_dim=None,
                 existing_vocab=None,
                 trainable=True,
                 name="embed"):

        if vocab_size is None and existing_vocab is None:
            raise ValueError("both `vocab_size` and `existing_vocab` are none")

        if existing_vocab is not None and not all(
                x is None for x in [vocab_size, embed_dim]):
            raise ValueError("When `existing_vocab` is provided, some of the "
                             "arguments should not be provided.")

        super(Embeddding, self).__init__(name=name)
        if existing_vocab is None:
            embed_dim = embed_dim or _embedding_dim(self._vocab_size)
        else:
            existing_vocab = tf.convert_to_tensor(
                existing_vocab, dtype=tf.float32)
            existing_vocab_shape = existing_vocab.get_shape().with_rank(2)
            existing_vocab_shape.assert_is_fully_defined()
            vocab_size, embed_dim = existing_vocab_shape.as_list()

        self._vocab_size = vocab_size
        self._embed_dim = embed_dim
        self._existing_vocab = existing_vocab
        self._trainable = trainable
        self._initializer = utils.create_linear_initializer(vocab_size)

    def _build(self, ids):
        """Lookup embeddings."""
        if self._existing_vocab is None:
            self._embeddings = tf.get_variable(
                "embeddings",
                shape=[self._vocab_size, self._embed_dim],
                dtype=tf.float32,
                initializer=self._initializer,
                trainable=self._trainable)
        else:
            self._embeddings = tf.get_variable(
                "embeddings",
                dtype=tf.float32,
                initializer=self._existing_vocab,
                trainable=self._trainable)

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
    """Module for embdding tokens using TF-Hub ELMO
        
       More information regarding the ELMO Module can be found in
       https://alpha.tfhub.dev/google/elmo/2
    """
    
    ELMO_URL = "https://tfhub.dev/google/elmo/2"

    def __init__(self, trainable=False, name="elmo_embed"):
        super(TFHubElmoEmbedding, self).__init__(name=name)
        self._trainable = trainable
        self._elmo = tf_hub.Module(self.ELMO_URL, trainable=trainable)

    def _build(self, tokens_input, tokens_length):
        """Compute the ELMO embeddings

        Args:
            tokens_input: tf.string Tensor of [batch_size, max_length]
            tokens_length: tf.int32 Tensor of [batch_size]

        Returns:
            embeddings: weighted sum of 3 layers (from ELMO model)
                        [batch_size, max_length, 1024]
        """
        if tokens_input.dtype != tf.string:
            raise TypeError("`tokens_input` must be tf.string")

        embeddings = self._elmo(
            inputs={"tokens": tokens_input,
                    "sequence_len": tokens_length},
            signature="tokens",
            as_dict=True)["elmo"]

        return embeddings

    def _clone(self, name):
        return type(self)(trainable=self._trainable, name=name)
