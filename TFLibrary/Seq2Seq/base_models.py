from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from collections import namedtuple


EncoderHParams = namedtuple("EncoderHParams",
    ("unit_type", "num_units", "num_layers",
    "dropout_rate", "num_residual_layers"))


class BaseEncoder(object):
    """Abstract encoder class.
       Implementations must define the following abstract methods:
       -`build`
       -`encode`
    """
    __metaclass__ = abc.ABCMeta


    @abc.abstractmethod
    def build(self, hparams, is_training=True):
        """Builder method for BaseEncoder.
        Args:
          hparams: An HParams object containing model hyperparameters.
          is_training: Whether or not the model is being used for training.
        """
        pass

    @abc.abstractmethod
    def encode(self, sequence, sequence_length):
        """Encodes input sequences into a precursors for latent code `z`.
        Args:
           sequence: Batch of sequences to encode.
           sequence_length: Length of sequences in input batch.
        Returns:
           outputs: Raw outputs to parameterize the prior distribution in
              MusicVae.encode, sized `[batch_size, N]`.
        """
        pass
