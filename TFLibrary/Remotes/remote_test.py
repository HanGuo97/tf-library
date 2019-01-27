import os
from absl.testing import absltest
from TFLibrary.Remotes import remote
from TFLibrary.Metrics import metrics
from TFLibrary.utils.misc_utils import read_text_file_utf8

TESTFILES_BASE_DIR = "../Metrics/testdata"


def start_server():
    func = metrics.SentenceROUGE()
    server = remote.ZMQServer(
        func=func, address="tcp://*:3000")
    server.start()


class MetricsTest(absltest.TestCase):
    def testZMQServerClient(self):
        func = metrics.SentenceROUGE()
        client = remote.ZMQClient(address="tcp://localhost:3000")
        self.assertScoresEqual(
            func=func,
            client=client,
            referenece_fname="target.0.txt",
            prediction_fname="prediction.0.txt",
            print_score=True)

    def assertScoresEqual(self,
                          func,
                          client,
                          referenece_fname,
                          prediction_fname,
                          print_score=False,
                          expect_equal=True):

        prediction_fname = os.path.join(TESTFILES_BASE_DIR, prediction_fname)
        referenece_fname = os.path.join(TESTFILES_BASE_DIR, referenece_fname)

        expected = func(
            prediction=read_text_file_utf8(prediction_fname)[0],
            reference=read_text_file_utf8(referenece_fname)[0])

        actual = client(
            prediction=read_text_file_utf8(prediction_fname)[0],
            reference=read_text_file_utf8(referenece_fname)[0])

        if print_score:
            print("%.1f " % actual, end="", flush=True)

        if expect_equal:
            self.assertEqual(expected, actual)
        else:
            self.assertNotEqual(expected, actual)


if __name__ == "__main__":
    absltest.main()
