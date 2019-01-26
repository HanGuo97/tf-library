from TFLibrary.Metrics.utils import bleu
from TFLibrary.Metrics.utils import rouge
from TFLibrary.Metrics import metrics_base


class SentenceBLEU(metrics_base.SentenceMetrics):
    def __init__(self, max_order=4, smooth=False, scale=False):
        super(SentenceBLEU, self).__init__()
        self._max_order = max_order
        self._smooth = smooth
        self._scale = scale

    def _call(self, prediction, reference):
        processed_reference = [[reference.split(" ")]]
        processed_prediction = [prediction.split(" ")]
        # bleu_score, precisions, bp, ratio,
        # translation_length, reference_length
        bleu_score, _, _, _, _, _ = bleu.compute_bleu(
            processed_reference,
            processed_prediction,
            max_order=self._max_order,
            smooth=self._smooth)

        if self._scale:
            return bleu_score * 100
        else:
            return bleu_score


class SentenceROUGE(metrics_base.SentenceMetrics):
    def __init__(self, scale=False):
        super(SentenceROUGE, self).__init__()
        self._scale = scale

    def _call(self, prediction, reference):
        rouge_score_map = rouge.rouge([prediction], [reference])
        
        if self._scale:
            return rouge_score_map["rouge_l/f_score"] * 100
        else:
            return rouge_score_map["rouge_l/f_score"]
