import os
import shutil
import oyaml as yaml
from glob import glob
import tensorflow as tf
from collections import Counter
from TFLibrary.Tuner import tuner
from TFLibrary.utils import misc_utils


LOGDIR = "./TunerTest"


class TunerTest(tf.test.TestCase):
    def tearDown(self):
        if os.path.exists(LOGDIR):
            shutil.rmtree(LOGDIR)

    def testOverlappingSpace(self):
        executable = ["echo hello"]
        with open("tests/space_overlap.yaml") as f:
            configs = list(yaml.load_all(f))
        
        for config in configs:
            with self.assertRaises(ValueError) as e:
                tuner.HparamOptimizer(
                    logdir=LOGDIR,
                    config=config,
                    executable=executable)
            print("\t", e.exception)

    def testInvalidConfigEntries(self):
        executable = ["echo hello"]
        with open("tests/invalid_entries.yaml") as f:
            configs = list(yaml.load_all(f))
        
        for config in configs[1:]:
            with self.assertRaises(ValueError) as e:
                tuner.HparamOptimizer(
                    logdir=LOGDIR,
                    config=config,
                    executable=executable)

            print("\t", e.exception)


        # This should pass
        tuner.HparamOptimizer(
            logdir=LOGDIR,
            config=configs[0],
            executable=executable)

    def testGridSearch(self):
        with open("tests/grid_search.yaml") as f:
            config = yaml.load(f)[0]
        
        param_space = config["paramSpace"]
        expected_outputs = []
        for a1 in param_space["gridArg1"]:
            for a2 in param_space["gridArg2"]:
                for a3 in param_space["gridArg3"]:
                    for a4 in param_space["gridArg4"]:
                        for a5 in param_space["gridArg5"]:
                            for a6 in param_space["gridArg6"]:
                                expected_outputs.append(
                                    "Arg1=%s Arg2=%s Arg3=%s "
                                    "Arg4=%s Arg5=%s Arg6=%s" %
                                    (a1, a2, a3, a4, a5, a6))

        grid_search_opt = tuner.HparamOptimizer(
            logdir=LOGDIR,
            config_file="tests/grid_search.yaml",
            executable_file="tests/grid_search.sh",
            print_command=False)

        nested_grid_search_opt = tuner.HparamOptimizer(
            logdir=LOGDIR,
            config_file="tests/grid_search_nested.yaml",
            executable_file="tests/grid_search.sh",
            print_command=False)

        grid_search_multigpu_opt = tuner.HparamOptimizer(
            logdir=LOGDIR,
            gpus="0,1,3".split(","),
            config_file="tests/grid_search.yaml",
            executable_file="tests/grid_search.sh",
            print_command=False)

        nested_grid_search_multigpu_opt = tuner.HparamOptimizer(
            logdir=LOGDIR,
            gpus="0,1,3,2,5,7,11,15".split(","),
            config_file="tests/grid_search_nested.yaml",
            executable_file="tests/grid_search.sh",
            print_command=False)

        for opt in [grid_search_opt,
                    nested_grid_search_opt,
                    grid_search_multigpu_opt,
                    nested_grid_search_multigpu_opt]:
            
            if not os.path.exists(LOGDIR):
                os.makedirs(LOGDIR)
            
            opt.tune()
            actual_outputs = [
                misc_utils.read_text_file(fname)[0]
                for fname in glob("TunerTest/*.log")]
            self.assertEqual(sorted(expected_outputs),
                             sorted(actual_outputs))
            
            shutil.rmtree(LOGDIR)

    def testBayesian(self):
        def evaluation_fn(hparams):
            fname = "TunerTest/%s_%s_%s_%s_%s.observ" % (
                hparams["bayesianArg1"],
                hparams["bayesianArg2"],
                hparams["bayesianArg3"],
                hparams["bayesianArg4"],
                hparams["bayesianArg5"])
            return -float(misc_utils.read_text_file(fname)[0])

        def _compute_expected_value():
            # - max(vals) == min(vals)
            return - max([
                float(misc_utils.read_text_file(fname)[0])
                for fname in glob("TunerTest/*.observ")])
            
        bayesian_opt = tuner.HparamOptimizer(
            logdir=LOGDIR,
            config_file="tests/bayesian_min.yaml",
            executable_file="tests/bayesian_min.sh",
            evaluation_func=evaluation_fn,
            print_command=False)

        nested_bayesian_opt = tuner.HparamOptimizer(
            logdir=LOGDIR,
            config_file="tests/bayesian_min_nested.yaml",
            executable_file="tests/bayesian_min.sh",
            evaluation_func=evaluation_fn,
            print_command=False)

        bayesian_multigpu_opt = tuner.HparamOptimizer(
            logdir=LOGDIR,
            gpus="0,1,3".split(","),
            config_file="tests/bayesian_min.yaml",
            executable_file="tests/bayesian_min.sh",
            evaluation_func=evaluation_fn,
            print_command=False)

        nested_grid_search_multigpu_opt = tuner.HparamOptimizer(
            logdir=LOGDIR,
            gpus="0,1,3,2,5,7,11,15".split(","),
            config_file="tests/bayesian_min_nested.yaml",
            executable_file="tests/bayesian_min.sh",
            evaluation_func=evaluation_fn,
            print_command=False)


        for opt in [bayesian_opt,
                    nested_bayesian_opt,
                    bayesian_multigpu_opt,
                    nested_grid_search_multigpu_opt]:
            
            if not os.path.exists(LOGDIR):
                os.makedirs(LOGDIR)
            
            actual_outputs = opt.tune()
            expected_outputs = _compute_expected_value()
            self.assertEqual(expected_outputs, actual_outputs)
            
            shutil.rmtree(LOGDIR)

    def testFused(self):
        def evaluation_fn(hparams):
            fname = "TunerTest/%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s.observ" % (
                hparams["gridArg1"],
                hparams["gridArg2"],
                hparams["gridArg3"],
                hparams["gridArg4"],
                hparams["gridArg5"],
                hparams["gridArg6"],
                hparams["bayesianArg1"],
                hparams["bayesianArg2"],
                hparams["bayesianArg3"],
                hparams["bayesianArg4"],
                hparams["bayesianArg5"])
            return -float(misc_utils.read_text_file(fname)[0])

        def _compute_expected_value():
            # - max(vals) == min(vals)
            return - max([
                float(misc_utils.read_text_file(fname)[0])
                for fname in glob("TunerTest/*.observ")])

        with open("tests/fused.yaml") as f:
            configs = list(yaml.load_all(f))

        # Two documents have the same paramSpace
        param_space = configs[0][0]["paramSpace"]
        expected_outputs_1 = []
        for a1 in param_space["gridArg1"]:
            for a2 in param_space["gridArg2"]:
                for a3 in param_space["gridArg3"]:
                    for a4 in param_space["gridArg4"]:
                        for a5 in param_space["gridArg5"]:
                            for a6 in param_space["gridArg6"]:
                                expected_outputs_1.append(
                                    "gArg1=%s gArg2=%s gArg3=%s "
                                    "gArg4=%s gArg5=%s gArg6=%s" %
                                    (a1, a2, a3, a4, a5, a6))

        for config in configs:
            for gpus in [None, "0,1,3,2,5,7,9,11,15,17".split(",")]:
                opt = tuner.HparamOptimizer(
                    logdir=LOGDIR,
                    gpus=gpus,
                    config=config,
                    executable_file="tests/fused.sh",
                    evaluation_func=evaluation_fn,
                    print_command=False)

                if not os.path.exists(LOGDIR):
                    os.makedirs(LOGDIR)

                actual_outputs_2 = opt.tune()
                actual_outputs_1 = [
                    misc_utils.read_text_file(fname)[0]
                    for fname in glob("TunerTest/*.log")]

                # There should be 10 copies of set(actual_outputs_1)
                # Since each gridSearch runs over 10 BayesianOpt
                actual_counter = Counter(actual_outputs_1)
                self.assertEqual({10}, set(list(actual_counter.values())))
                self.assertEqual(sorted(expected_outputs_1),
                                 sorted(list(actual_counter.keys())))

                expected_outputs_2 = _compute_expected_value()
                self.assertEqual(expected_outputs_2, actual_outputs_2)

                shutil.rmtree(LOGDIR)


if __name__ == "__main__":
    tf.test.main()
