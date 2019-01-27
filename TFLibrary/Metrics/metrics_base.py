"""Abstract Class for Metrics"""

import os
from TFLibrary.utils.misc_utils import read_text_file_utf8


class Metrics(object):
    """Base Class for Evaluation Metrics
    
    Todo: Make this Registrable
    https://github.com/allenai/allennlp/blob/master/allennlp/common/registrable.py
    """
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class SentenceMetrics(Metrics):
    """Metrics on pair of sentences"""
    def __call__(self, prediction, reference):
        if not isinstance(prediction, str):
            raise TypeError("`prediction` must be String, "
                            "found ", type(prediction))

        if not isinstance(reference, str):
            raise TypeError("`reference` must be String, "
                            "found ", type(reference))

        return self._call(prediction=prediction,
                          reference=reference)

    def _call(self, prediction, reference):
        raise NotImplementedError


class CorpusMetrics(Metrics):
    """Metrics on corpus"""
    def _maybe_load_from_file(self, data, filename):
        if data is None:
            if not filename:
                raise ValueError("When data is None, "
                                 "file name must be provided")

            if not os.path.exists(filename):
                raise ValueError("Filename not exists")

            data = read_text_file_utf8(filename)
        
        return data

    def __call__(self,
                 predictions=None,
                 prediction_fname=None,
                 references=None,
                 reference_fname=None):

        predictions = self._maybe_load_from_file(
            data=predictions,
            filename=prediction_fname)

        references = self._maybe_load_from_file(
            data=references,
            filename=reference_fname)

        if not isinstance(predictions, (list, tuple)):
            raise TypeError("`predictions` should be list-like")

        if not isinstance(references, (list, tuple)):
            raise TypeError("`references` should be list-like")

        return self._call(predictions=predictions,
                          references=references)

    def _call(self, predictions, references):
        raise NotImplementedError
