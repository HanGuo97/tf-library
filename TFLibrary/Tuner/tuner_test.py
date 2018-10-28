import os
import shutil
import oyaml as yaml
from glob import glob
import tensorflow as tf
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





if __name__ == "__main__":
    tf.test.main()
