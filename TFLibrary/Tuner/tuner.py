import os
import copy
import oyaml as yaml
from tqdm import tqdm
from collections import namedtuple
from collections import OrderedDict
from TFLibrary.utils import misc_utils
from TFLibrary.Tuner import utils
from TFLibrary.Tuner import reduce_ops
from TFLibrary.Tuner import runners as runner_ops
from TFLibrary.Tuner import optimizers as optimizer_ops

FeedbackCollection = namedtuple(
    "FeedbackCollection",
    ("Observation", "Feedback"))

# Keys that must present in the config
# and corresponding allowed Values.
# `None` Values means all entries are allowed
MANDATORY_KEYS_AND_VALUES = {
    # `optimizerType` specifies the type of optimizer
    "optimizerType": ["gridSearch", "bayesianMin"],
    # `maxExperiments` specifies the maximum optmizing step
    # only useful in Bayesian optimizer setting
    "maxExperiments": None,
    # Running parallel trials has the benefit of reducing the
    # time the training job takes (real time—the total processing
    # time required is not typically changed). However, running in
    # parallel can reduce the effectiveness of the tuning job overall.
    # That is because hyperparameter tuning uses the results of
    # previous trials to inform the values to assign to the
    # hyperparameters of subsequent trials. When running in parallel,
    # some trials start without having the benefit of the results
    # of any trials still running.
    "maxParallelExperiments": None,
    # `reduceOp` defines how values from one optimizer
    # are aggregated and backprop to the optimizer one level up
    "reduceOp": ["min", None],
    # hyper-parameter search space
    # should be a list of dictionaries
    "paramSpace": None
}

# Optional Keys and Entries
OPTIONAL_KEYS_AND_VALUES = {
    # additional arguments passed to optimizer
    "optimizerArgs": None,
}


class HparamOptimizer(object):
    """ Hyperparameter Optimizer

    Optimizer takes as input a configuration file specifying the
    hyperparameter search space and a executive file that takes
    hyperparameters and return some values. The tuner then will
    optimize the values by searching over the space, using either
    grid search and/or Bayesian optimization approaches.

    Tests:
        - Given list of Hparms to singleGpuTuner

    Todos:
        - Current parallelization implementation only parallizes
            at the lowest level. It will be useful for GPUs parallel
            by level (multiGPU on GridSearch, single on Bayesian etc)
        - Returns of Optimizer.observe could be helpful
        - Add `oyaml` to REQUIREMENTS
    """
    def __init__(self,
                 logdir,
                 gpus=None,
                 config=None,
                 config_file=None,
                 executable=None,
                 executable_file=None,
                 evaluation_func=None,
                 print_command=False):
        """Create a Optimizer

        Args:
            Logdir:
                String
                Directory for saving stuff
            gpus:
                List of Strings
                GPU-IDs to be run in parallel
            config, config_file:
                OrderedDict or String
                Parameter YAML or the directory to the YAML file
                If both are specified, config has higher priority
            executable, executable_file:
                List or String
                List of Shell commands or the directory to the shell script
                If both are specified, executable has higher priority
            evaluation_func:
                Callable(hparams_instance)
                The function to be called after executing
                the tunee to collect evaluation observations
                in arbitrary structure
        """
        if gpus and not isinstance(gpus, (list, tuple)):
            raise TypeError("`gpus` must be list or tuple")

        # if config and not isinstance(config, (tuple, list)):
        #     raise TypeError("`config` must be `OrderedDict`")

        # if executable and not isinstance(executable, (tuple, list)):
        #     raise TypeError("`executable` must be `tuple` or `list`")

        if evaluation_func is None:
            evaluation_func = lambda _hparams: None

        if not callable(evaluation_func):
            raise TypeError("`evaluation_func` must be callable")

        # read the YAML file into OrderedDict
        # using OrderedDict to prevent accident re-ordering
        # of key-val pairs in later stages
        if config is None:
            with open(config_file) as f:
                config = yaml.load(f)
        
        if executable is None:
            with open(executable_file) as f:
                executable = [d.strip("\n") for d in f.readlines()]

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        self._gpus = gpus
        self._logdir = logdir
        self._config = config
        self._executable = executable
        self._print_command = print_command
        self._evaluation_func = evaluation_func

        # `histories` caches the evaluation histories
        self._histories = []

        # `runner` is used for executing experiments
        self._runner = None

        # restoration will overwrite setup
        self._restore_or_setup()
        self._validate_configuration()
        self._validate_assumtpions()

    @property
    def max_parallel(self):
        if self._gpus is not None:
            return len(self._gpus)
        return None

    def _validate_configuration(self):
        """Check the format of the configuration:
        
        0. Check if the nested length exceeds 2
        1. Check whether all keys and values in `config` are valid
        2. Check whether the space of multiple optimizers overlap
        """
        # Currently only nested optimizers of depth 2 is tested
        if len(self._config) != 2:
            raise NotImplementedError(
                "Only Nested config of depth within 2 is tested")

        # Check the validity of `config` entries
        for config in self._config:
            # check for args and entries
            for key, vals in MANDATORY_KEYS_AND_VALUES.items():
                if key not in config:
                    raise ValueError(
                        "Mandatory key `%s` not in config" % key)
                if vals is not None:
                    if config[key] not in vals:
                        raise ValueError(
                            "Value `%s` of Mandatory Key `%s` "
                            "not in %s" % (config[key], key, vals))

            for key, vals in OPTIONAL_KEYS_AND_VALUES.items():
                if key not in config:
                    continue
                if vals is not None:
                    if config[key] not in vals:
                        raise ValueError(
                            "Value `%s` of Optional Key `%s` "
                            "not in %s" % (config[key], key, vals))

        # check whether the space keys overlap
        # this implementation is inefficient in compuation
        # but should be more readable
        joint_space_keys = set()
        for config in self._config:
            space_keys = set(config["paramSpace"])
            key_intersect = joint_space_keys.intersection(space_keys)
            if key_intersect:
                raise ValueError("Space overlapping between "
                                 "configurations %s" % key_intersect)

            joint_space_keys.update(space_keys)


    def _restore_or_setup(self):
        """Restore or Setup the Optimizer"""
        # currently not supported
        self._restore()
        self._setup()

    def _restore(self):
        # currently not supported
        pass

    def _save(self):
        # currently not supported
        pass

    def _setup(self):
        """Setting up:
        
        - `runner` takes experiments (i.e. executable files) and
            execute them, in sequence or in parallel.
        """
        # Build Runner
        if self.max_parallel is None:
            raise NotImplementedError("Non-Parallel NotImplemented")
        else:
            runner = runner_ops.SyncMultiGPURunner(
                gpus=self._gpus,
                logdir=self._logdir,
                print_command=self._print_command)
            print("SyncMultiGPURunner: \t Num GPUs=%d" % runner.max_processes)

        self._runner = runner

    def _execute_and_evaluate(self, hparams):
        """Execute and Evaluate Hparams, potentially in parallel."""
        if self.max_parallel is None:
            if not isinstance(self._runner, runner_ops.BasicRunner):
                raise TypeError("runner must be `BasicRunner`")

            # Insert `hparams` into `_executable`
            experiment = utils._create_experiment(
                executable=self._executable,
                hparams_instance=hparams)

            # Run and Evaluate
            self._runner.run(experiment)
            return self._evaluation_func(hparams)

        else:
            if not isinstance(self._runner, runner_ops.SyncMultiGPURunner):
                raise TypeError("runner must be `SyncMultiGPURunner`")

            # Insert multiple `hparams` into multiple `_executable`
            experiments = [
                utils._create_experiment(
                    executable=self._executable,
                    hparams_instance=hparam)
                for hparam in hparams]

            # Run and Evaluate
            self._runner.run(experiments)
            return [self._evaluation_func(hparam) for hparam in hparams]

    def tune(self):
        # Setting up Progress Bar, which will be used
        # inside `self._tune()`
        self._temp_pbar = None

        try:
            to_return = self._tune()
        finally:
            # Ensure the pbar is closed
            if self._temp_pbar is not None:
                self._temp_pbar.close()

        return to_return

    def _validate_assumtpions(self):
        # There should be only two optimizers
        if len(self._config) != 2:
            raise ValueError("There must be two and only two optimizers")

        if (self._config[0]["optimizerType"] != "gridSearch" and
                self._config[1]["optimizerType"] != "bayesianMin"):
            raise ValueError("Must consist of gridSearch + bayesianMin")

        total_parallel_experiments = (
            self._config[0]["maxParallelExperiments"] *
            self._config[1]["maxParallelExperiments"])
        if total_parallel_experiments != self.max_parallel:
            raise ValueError(
                "configs have `maxParallelExperiments` = %d x %d, #GPUs = %d"
                % (self._config[0]["maxParallelExperiments"],
                   self._config[1]["maxParallelExperiments"],
                   self.max_parallel))

        
    def _tune(self):
        """Tune the model Using GridSearch and Bayesian Optimization.

        Let `optimizer_A` and `optimizer_B` as GridOpt and BayesOpt,
        where optimizer_A is on first level and optimizer_B is on the second
        level. We can view the combinations of optimizer_A and optimizer_B
        in a table:
        
        (A0, B0) o--o (A1, B0) o--o (A2, B0)
           |             |             |
           |             |             |
           |             |             |
        (A0, B1) o--o (A1, B1) o--o (A2, B1)
           |             |             |
           |             |             |
           |             |             |
        (A0, B2) o--o (A1, B2) o--o (A2, B2)
        
        Note that because of the hierarchical structure, there is a single
        optimizer_A, but multiple independent optimizer_B's. We will assume:
            1) The max experiments is the same for all optimizer_B.
            2) optimizer_A's param decisions are independent, while
                optimizer_B's decisions are, thus we should minimize
                parallelization on optimizer_B while maximize parallelzation
                on optimizer_A.
        """
        config_A, config_B = self._config

        max_parallel_A = config_A["maxParallelExperiments"]
        max_parallel_B = config_B["maxParallelExperiments"]
        (optimizer_A, reduce_func_A, max_experiments_A) = (
            _build_optimizer_from_config(copy.deepcopy(config_A)))
        
        feedback_histories_A = []
        observation_histories_A = []
        feedback_histories_B = [[] for _ in range(max_experiments_A)]
        observation_histories_B = [[] for _ in range(max_experiments_A)]
        for i in range(0, max_experiments_A, max_parallel_A):
            # `max_experiments_A - i` is the remaining experiments
            num_to_query_A = min(max_parallel_A, max_experiments_A - i)
            params_A_batch = optimizer_A.query(num_to_query_A)
            
            # Assemble optimizers_B
            # Assume all optmizers_B have the same `maxExperiments`
            # and reduce_func, thus simply using those from last one
            optimizer_Bs = []
            for _ in range(num_to_query_A):
                (optimizer_B, reduce_func_B, max_experiments_B) = (
                    _build_optimizer_from_config(copy.deepcopy(config_B)))
                optimizer_Bs.append(optimizer_B)


            for j in range(0, max_experiments_B, max_parallel_B):
                # `max_experiments_B - j` is the remaining experiments
                num_to_query_B = min(max_parallel_B, max_experiments_B - j)
                params_B_batchs = [
                    optB.query(num_to_query_B) for optB in optimizer_Bs]
                
                # params_A_batch = [num_to_query_A]
                # params_B_batchs = [num_to_query_A, num_to_query_B]
                # params_to_run = [num_to_query_A x num_to_query_B]
                params_to_run = []
                for pA, pBs in zip(params_A_batch, params_B_batchs):
                    params_to_run.extend([
                        misc_utils.merge_ordered_dicts(pA, pB)
                        for pB in pBs])

                # Run the command, return [num_to_query_A x num_to_query_B]
                observations_B = self._execute_and_evaluate(params_to_run)

                # Create progress bar if not existed
                if self._temp_pbar is None:
                    _num_experiments = max_experiments_A * max_experiments_B
                    self._temp_pbar = tqdm(total=_num_experiments)

                # Update the Progress Bar
                self._temp_pbar.update(len(params_to_run))
                
                # Update all optimizer_Bs
                for z, optB in enumerate(optimizer_Bs):
                    _start_index = z * num_to_query_B
                    _end_index = (z + 1) * num_to_query_B
                    _observations = observations_B[_start_index: _end_index]
                    feedback_B = optB.observe(
                        params=params_B_batchs[z],
                        observation=_observations)
                    
                    feedback_histories_B[i + z].append(feedback_B)
                    observation_histories_B[i + z].extend(_observations)
            
            
            # [num_to_query_A, num_to_query_B] to [num_to_query_A]
            observations_A = [
                reduce_func_B(obhB) for obhB in
                observation_histories_B[i: i + num_to_query_A]]
            
            feedback_A = optimizer_A.observe(
                params=params_A_batch,
                observation=observations_A)
            feedback_histories_A.append(feedback_A)
            observation_histories_A.extend(observations_A)
        
        return (reduce_func_A(observation_histories_A),
                (feedback_histories_A, feedback_histories_B))


def _build_optimizer_from_config(config):
    # Create a new Optimizer instance
    if "optimizerArgs" not in config:
        config["optimizerArgs"] = {}

    # Build Optimizer
    if config["optimizerType"] == "gridSearch":
        optimizer = optimizer_ops.GridSearchOptimizer(
            param_space=config["paramSpace"],
            **config["optimizerArgs"])
    elif config["optimizerType"] == "bayesianMin":
        optimizer = optimizer_ops.SkoptBayesianMinOptimizer(
            param_space=config["paramSpace"],
            **config["optimizerArgs"])
    else:
        raise ValueError(
            "`optimizer_type` `%s` "
            "not recognized" % config["optimizerType"])

    # Build Reduce Function for aggregating observations
    if config["reduceOp"] is None:
        reduce_func = reduce_ops.no_op
    if config["reduceOp"] == "mean":
        reduce_func = reduce_ops.reduce_mean
    if config["reduceOp"] == "min":
        reduce_func = reduce_ops.reduce_min

    # Max steps to limit run time
    if (optimizer.num_iterations is not None and
            config["maxExperiments"] is not None):
        num_iterations = min(optimizer.num_iterations,
                             config["maxExperiments"])

    elif optimizer.num_iterations is not None:
        num_iterations = optimizer.num_iterations

    elif config["maxExperiments"] is not None:
        num_iterations = config["maxExperiments"]


    return optimizer, reduce_func, num_iterations
