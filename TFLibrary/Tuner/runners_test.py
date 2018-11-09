import os
import shutil
from glob import glob
import tensorflow as tf
from TFLibrary.utils import misc_utils
from TFLibrary.Tuner import runners as runner_ops


LOGDIR = "./TunerTest"
COMMAND = lambda i: ["echo test_ONE_%d" % (i),
                     "echo test_THREE_%d" % (i * 2),
                     "echo test_TWO_%d" % (i)]


class OptimizersTest(tf.test.TestCase):
    def testSyncMultiGPURunner(self):
        gpus = "0,1,2".split(",")
        runner = runner_ops.SyncMultiGPURunner(
            gpus=gpus, logdir=LOGDIR, print_command=True)
        runner.run([COMMAND(i) for i in range(len(gpus))])

        expected_outputs = [
            ["test_ONE_0", "test_THREE_0", "test_TWO_0"],
            ["test_ONE_1", "test_THREE_2", "test_TWO_1"],
            ["test_ONE_2", "test_THREE_4", "test_TWO_2"]]

        actual_outputs = [
            misc_utils.read_text_file(fname)
            for fname in sorted(glob(os.path.join(LOGDIR, "*.log")))]

        shutil.rmtree(LOGDIR)
        print("expected_outputs\n", expected_outputs)
        print("actual_outputs\n", actual_outputs)
        self.assertEqual(expected_outputs, actual_outputs)


if __name__ == "__main__":
    tf.test.main()
