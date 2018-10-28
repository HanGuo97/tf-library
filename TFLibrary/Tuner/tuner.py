import os
import copy
import oyaml as yaml
from tqdm import tqdm
from collections import namedtuple
from collections import OrderedDict
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
    # `maxOptimizeSteps` specifies the maximum optmizing step
    # only useful in Bayesian optimizer setting
    "maxOptimizeSteps": None,
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

    @property
    def num_parallel(self):
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
        if len(self._config) > 2:
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
        - `optimizer` observes the outputs of experiments and decide
            the next parameter space point to query.
        - `reduce_func` takes the set of observations from one optimizer,
            and reduce and passed them to another optimizer
        - `max_steps` the max number of optimization steps

        """
        # Build Runner
        if self.num_parallel is None:
            runner = runner_ops.BasicRunner(
                logdir=self._logdir,
                print_command=self._print_command)
        else:
            runner = runner_ops.MultiGPURunner(
                gpus=self._gpus,
                logdir=self._logdir,
                print_command=self._print_command)
            print("MultiGPURunner: \t Num GPUs=%d" % runner.num_gpus)

        self._runner = runner

    def _execute_and_evaluate(self, hparams):
        """Execute and Evaluate Hparams, potentially in parallel."""
        if self.num_parallel is None:
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
            if not isinstance(self._runner, runner_ops.MultiGPURunner):
                raise TypeError("runner must be `MultiGPURunner`")

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

        
    def _tune(self):
        """Tuning

        The Optmizers are assumed to be:
        - independent: the optimizing trajectory of one optimizer
            does not change the trajectory of another optimizer

        """
        def recursive_loop(configs,
                           hparams=OrderedDict(),
                           cum_num_iterations=1):
            """Recursively loop over all optimizers.

            Args:
                configs: list of Optimizer configurations
                hparams: list of hparams
                pbar: `tqdm.trange()` from previous optimizer

            Since the number of optimizers are arbitrary, and nested. We
            use a recursive function to loop over all optimizers. For
            example, suppose we have three optimizers A, B, and C. A iterative
            implementation should be:

            ```python

            for i in range(lenA):
                observationsA = []
                hparamsA = optA.query()
                
                for j in range(lenB):
                    observationsB = []
                    hparamsB = optB.query()
                    
                    for k in range(lenC):
                        observationsC = []
                        hparamsC = optC.query()
                        
                        merged_hparams = merge_hparams(
                            hparamsA, hparamsB, hparamsC)
                        experiment = create_experiment(merged_hparams)
                        observation = run_and_evaluate(experiment)
                        
                        optC.observe(observation)
                        observationsC.append(observation)

                    reducedC = reduceFnC(observationsC)
                    optB.observe(reducedC)
                    observationsB.append(reducedC)

                reducedB = reduceFnC(observationsB)
                optA.observe(reducedB)
                observationsA.append(reducedB)

            reducedA = reduceFnC(observationsA)
            return observations
            ```

            Obviously, this is overly nested and non-flexible. Thus
            we instead following a recursive implementation. Note that
            in parallel setting, only the optimizer at the lowest level
            will produce multiple hparams, whereas all upper level optimizers
            runs as if non-parallel setting.

            Note that in nested settings, lower optimizer need to be
            re-initialized after each iteration in the higher level
            (since the optimizer will become exhausted). Thus at each
            level, we build a new instance of optimizer.
            """
            # When we reach the bottom level of optimizers,
            # we will actually run the model. Since we need to
            # know the joint hparams, during each recursive call
            # new hparams will be merged into hparams and passed
            # on to the next level -- and we will have the full hparams
            # at the bottom level. We also keep track of the total
            # iteration counts for visualizing the tuning progress.
            if not configs:
                # Create progress bar if not existed
                if self._temp_pbar is None:
                    self._temp_pbar = tqdm(total=cum_num_iterations)

                self._temp_pbar.update(
                    1 if self.num_parallel is None
                    else len(hparams))
                
                return self._execute_and_evaluate(hparams)
                
            # In parallel setting, the last optimizer
            # will run in parallel, whereas other optimizers
            # run in non-parallel setting
            num_parallel = (
                self.num_parallel
                # len(configs) == 1 indicates last opt
                if len(configs) == 1 else None)

            # Build Optimizer (essentially re-initialize it)
            optimizer, reduce_func, num_iterations = (
                _build_optimizer_from_config(configs[0]))

            # Merge the total iteration counts from
            # one level up with the `num_iterations` here
            new_cum_num_iterations = cum_num_iterations * num_iterations
            
            # Observations caches all evaluation performance
            # and will be aggregated and passed onto upper levels
            observations = []

            # Essentially equals to range(ceil(num_iterations / num_parallel)):
            for step in range(0, num_iterations, num_parallel or 1):
                # Make sure we don't exceed `cum_num_iterations`
                # This is possible under Bayesian + Parallel setting
                # e.g. there are in total 10 iterations, with 7 threads
                # then the system will overshoot
                num_to_query = (
                    num_parallel if num_parallel is None else
                    min(num_parallel, num_iterations - step))

                # Ask for the suggested hparams
                suggested_hparams = optimizer.query(num_to_query)

                # Merge the hparams from one level up
                # with the hparams at this level
                new_hparams = utils.merge_hparams(
                    hparams_1=copy.deepcopy(hparams),
                    hparams_2=copy.deepcopy(suggested_hparams))

                # Recursively loop over the optimizer one level downn
                # and pass new_hparams from this optimizer
                observation = recursive_loop(
                    configs=configs[1:], hparams=new_hparams,
                    cum_num_iterations=new_cum_num_iterations)
                
                # Update the optimizer
                feedback = optimizer.observe(
                    params=suggested_hparams,
                    observation=observation)
                
                if isinstance(observation, (list, tuple)):
                    if len(observation) != len(suggested_hparams):
                        raise ValueError(
                            "len(observation) %d != "
                            "len(suggested_hparams) %d"
                            % (len(observation), len(suggested_hparams)))

                    # Cache the observations
                    # In parallel setting, `observation` itself
                    # is also a list, so we should instead add the
                    # list into `observations` instead of appending
                    observations = observations + observation
                else:
                    # Cache the observations
                    # Here `observation` is just a scalar
                    observations.append(observation)

                # Save the history
                # Histories are used for debugging
                self._histories.append(FeedbackCollection(
                    Observation=observation, Feedback=feedback))
            
            # Aggregate all observations from the optimizer
            # and reduce them into a single scalar to
            # be fed into optimizer one level up
            return reduce_func(observations)

        final_observation = recursive_loop(self._config)

        return final_observation


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
            config["maxOptimizeSteps"] is not None):
        num_iterations = min(optimizer.num_iterations,
                             config["maxOptimizeSteps"])

    elif optimizer.num_iterations is not None:
        num_iterations = optimizer.num_iterations

    elif config["maxOptimizeSteps"] is not None:
        num_iterations = config["maxOptimizeSteps"]


    return optimizer, reduce_func, num_iterations
