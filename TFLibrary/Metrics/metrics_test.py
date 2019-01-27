import os
from absl.testing import absltest
from TFLibrary.Metrics import metrics
from TFLibrary.Metrics.utils import nmt_evaluation_utils
from TFLibrary.utils.misc_utils import read_text_file_utf8

TESTFILES_BASE_DIR = "./testdata"


class MetricsTest(absltest.TestCase):

    def setUp(self):
        print("In method", self._testMethodName)

    def testSentenceBLEU(self):
        for prediction_fname in ["h1", "h2"]:
            for referenece_fname in ["r1", "r2", "r3"]:
                self.assertScoresEqual(
                    metric="bleu",
                    referenece_fname=referenece_fname,
                    prediction_fname=prediction_fname,
                    print_score=True)

        self.assertScoresEqual(
            metric="bleu",
            referenece_fname="target.0.txt",
            prediction_fname="prediction.0.txt",
            print_score=True)

        self.assertScoresEqual(
            metric="bleu",
            referenece_fname="target.1.txt",
            prediction_fname="prediction.1.txt",
            print_score=True)

        self.assertScoresEqual(
            metric="bleu",
            referenece_fname="target.txt",
            prediction_fname="prediction.txt",
            print_score=True,
            expect_equal=False)

        # Add "\n"
        print()

    def testSentenceROUGE(self):
        for prediction_fname in ["h1", "h2"]:
            for referenece_fname in ["r1", "r2", "r3"]:
                self.assertScoresEqual(
                    metric="rouge",
                    referenece_fname=referenece_fname,
                    prediction_fname=prediction_fname,
                    print_score=True)
        
        self.assertScoresEqual(
            metric="rouge",
            referenece_fname="target.0.txt",
            prediction_fname="prediction.0.txt",
            print_score=True)

        self.assertScoresEqual(
            metric="rouge",
            referenece_fname="target.1.txt",
            prediction_fname="prediction.1.txt",
            print_score=True)

        self.assertScoresEqual(
            metric="rouge",
            referenece_fname="target.txt",
            prediction_fname="prediction.txt",
            print_score=True,
            expect_equal=False)

        # Add "\n"
        print()

    def assertScoresEqual(self,
                          metric,
                          referenece_fname,
                          prediction_fname,
                          print_score=False,
                          expect_equal=True):
        if metric == "bleu":
            scorer = metrics.SentenceBLEU(scale=True)
        elif metric == "rouge":
            scorer = metrics.SentenceROUGE(scale=True)
        else:
            raise ValueError

        ref_file = os.path.join(TESTFILES_BASE_DIR, referenece_fname)
        pred_file = os.path.join(TESTFILES_BASE_DIR, prediction_fname)

        expected = nmt_evaluation_utils.evaluate(
            ref_file=ref_file,
            trans_file=pred_file,
            metric=metric)

        actual = scorer(
            prediction=read_text_file_utf8(pred_file)[0],
            reference=read_text_file_utf8(ref_file)[0])

        if print_score:
            print("%.1f " % actual, end="", flush=True)

        if expect_equal:
            self.assertEqual(expected, actual)
        else:
            self.assertNotEqual(expected, actual)


if __name__ == "__main__":
    absltest.main()
