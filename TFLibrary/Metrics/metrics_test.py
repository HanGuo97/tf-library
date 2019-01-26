import os
import tensorflow as tf
from TFLibrary.Metrics import metrics
from TFLibrary.Metrics.utils import nmt_evaluation_utils
from TFLibrary.utils.misc_utils import read_text_file_utf8

TESTFILES_BASE_DIR = "./TFLibrary/Metrics/tests"


class MetricsTest(tf.test.TestCase):

    def setUp(self):
        print("In method", self._testMethodName)

    def testSentenceBLEU(self):
        scorer = metrics.SentenceBLEU(scale=True)
        
        for hname in ["h1", "h2"]:
            for rname in ["r1", "r2", "r3"]:
                ref_file = os.path.join(TESTFILES_BASE_DIR, rname)
                trans_file = os.path.join(TESTFILES_BASE_DIR, hname)

                expected = nmt_evaluation_utils.evaluate(
                    ref_file=ref_file,
                    trans_file=trans_file,
                    metric="bleu")

                actual = scorer(
                    reference=read_text_file_utf8(ref_file)[0],
                    output=read_text_file_utf8(trans_file)[0])

                self.assertEqual(expected, actual)

    def testSentenceROUGE(self):
        scorer = metrics.SentenceROUGE(scale=True)
        
        for hname in ["h1", "h2"]:
            for rname in ["r1", "r2", "r3"]:
                ref_file = os.path.join(TESTFILES_BASE_DIR, rname)
                trans_file = os.path.join(TESTFILES_BASE_DIR, hname)

                expected = nmt_evaluation_utils.evaluate(
                    ref_file=ref_file,
                    trans_file=trans_file,
                    metric="rouge")

                actual = scorer(
                    reference=read_text_file_utf8(ref_file)[0],
                    output=read_text_file_utf8(trans_file)[0])

                self.assertEqual(expected, actual)


if __name__ == "__main__":
    tf.test.main()
