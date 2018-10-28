import copy
import itertools
from collections import OrderedDict, deque
from skopt import Optimizer as skoptOptimizer
from skopt.space import space as skopt_space



class Optimizer(object):
    """Base Class for Optimizers

    Optimizer is used to optimize a black box function
    over a set of pre-defined parameter search space.

    Methods:
        query (n: Integer):
            Query the optimizer to return the next
            set of parameters to evaluate. When
            `n` is not None, multiple instances
            will be returned.

        observe (params: Dict, value: Float)
            Inform the optimizer the value of the
            provided parameters.

        save, load (fname: string):
            Save or Load Optimizer from `fname`.
    
    
    Attributes:
        num_iterations: Integer
            Number of total search iterations,
            `None` if unlimited
    """
    def __init__(self, param_space):
        if not isinstance(param_space, OrderedDict):
            raise TypeError(
                "`param_space` must be an `OrderedDict`, "
                "but found to be %s" % type(param_space))

        self._param_space = param_space

    @property
    def num_iterations(self):
        raise NotImplementedError

    def query(self, n=None):
        raise NotImplementedError

    def observe(self, params, observation):
        raise NotImplementedError

    def save(self, fname):
        raise NotImplementedError

    def load(self, fname):
        raise NotImplementedError


class GridSearchOptimizer(Optimizer):
    def __init__(self, param_space, **kwarg):
        super(GridSearchOptimizer, self).__init__(
            param_space=param_space)
        instances = self._generate_hparam_instances(param_space)
        instance_queue = deque(instances)

        self._instances = instances
        self._instance_queue = instance_queue

    @property
    def num_iterations(self):
        return len(self._instance_queue)

    def query(self, n=None):
        if not self._instance_queue:
            raise ValueError("Optimizer Search is exhausted")

        # Fetch one hparams
        if n is None:
            return self._instance_queue.popleft()

        # Fetch multiple hparms
        hparams_instances = []
        for _ in range(n):
            # When the queue is exhausted
            if not self._instance_queue:
                break
            # Otherwise keep popping
            hparams_instance = self._instance_queue.popleft()
            hparams_instances.append(hparams_instance)

        return hparams_instances

    def observe(self, params, observation):
        pass

    def _generate_hparam_instances(self, d):
        # generate all combinations of dictionary values, unnamed
        value_collections = itertools.product(*d.values())
        # map the combination of values into a named dictionary
        # using OrderedDict simply to be consistent, but dict() also works
        hparam_collections = [OrderedDict((k, v)
                                          for k, v in zip(d.keys(), vals))
                              for vals in value_collections]
        return hparam_collections


class SkoptBayesianMinOptimizer(Optimizer):
    def __init__(self, param_space, **kwarg):
        super(SkoptBayesianMinOptimizer, self).__init__(
            param_space=param_space)
        self._dimensions = self._space_to_dimensions(
            # This function will modify dicts, so we
            # only use the copied version
            copy.deepcopy(param_space))
        self._optimizer = skoptOptimizer(
            dimensions=self._dimensions, **kwarg)

    @property
    def num_iterations(self):
        return None

    def query(self, n=None):
        # Optimizer returns a list of suggested params
        # convert them into dictionaries
        params_list = self._optimizer.ask(n_points=n)

        def _toOrderedDict(plist):
            pdict = OrderedDict()
            for dimension, params in zip(self._dimensions, plist):
                pdict[dimension.name] = params
            return pdict

        if n is None:
            return _toOrderedDict(params_list)

        return [_toOrderedDict(pl) for pl in params_list]

    def observe(self, params, observation):
        # In non-parallel setting, Params are OrderedDict
        if isinstance(params, OrderedDict):
            params_list = list(params.values())
            return self._optimizer.tell(
                x=params_list, y=observation)
            
        # In parallel setting, Params are list of OrderedDict
        if (isinstance(params, (list, tuple)) and
                isinstance(params[0], OrderedDict)):

            params_list = [list(p.values()) for p in params]
            return self._optimizer.tell(
                x=params_list, y=observation)

        raise TypeError(
            "In non-parallel setting, Params should be OrderedDict, "
            "and in parallel setting, Params should be list of OrderedDict ,"
            "found %s" % type(params))

    def _space_to_dimensions(self, param_space):
        dimensions = []

        # Format: Dict of Dicts
        # {ParamName: {type: SomeType, ...Specs...},
        #  ParamName: {type: SomeType, ...Specs... ...}
        for param_name, specs_dict in param_space.items():
            # This will raise KeyError if `type` not in `specs_dict`
            # and `type` will be removed from specs_dict
            space_type = specs_dict.pop("type")

            if space_type == "categorical":
                # categories, prior=None, transform=None, name=None
                dimension = skopt_space.Categorical(
                    name=param_name, **specs_dict)
            elif space_type == "integer":
                # low, high, transform=None, name=None
                dimension = skopt_space.Integer(
                    name=param_name, **specs_dict)

            elif space_type == "real":
                # low, high, prior='uniform', transform=None, name=None
                dimension = skopt_space.Real(
                    name=param_name, **specs_dict)

            else:
                raise ValueError("type `%s` unrecognized" % space_type)

            dimensions.append(dimension)

        return dimensions
