import oyaml as yaml
import tensorflow as tf
import optimizers as optimizer_ops
from collections import OrderedDict
from skopt.space import space as skopt_space


def get_example_config():
    with open("tests/example.yaml") as f:
        config = yaml.load(f)

    return config


class OptimizersTest(tf.test.TestCase):
    def testGridSearchOptimizer(self):
        config = get_example_config()
        # First config corresponds to GridSearch
        param_space = config[0]["paramSpace"]
        optimizer = optimizer_ops.GridSearchOptimizer(param_space)

        # Query all params
        actual_params = []
        for _ in range(optimizer.num_iterations):
            actual_params.append(optimizer.query())

        expected_params = []
        for a1 in param_space["gridArg1"]:
            for a2 in param_space["gridArg2"]:
                for a3 in param_space["gridArg3"]:
                    for a4 in param_space["gridArg4"]:
                        for a5 in param_space["gridArg5"]:
                            for a6 in param_space["gridArg6"]:
                                expected_params.append(OrderedDict([
                                    ("gridArg1", a1),
                                    ("gridArg2", a2),
                                    ("gridArg3", a3),
                                    ("gridArg4", a4),
                                    ("gridArg5", a5),
                                    ("gridArg6", a6)],
                                ))

        self.assertEqual(expected_params, actual_params)
        self.assertEqual(expected_params, optimizer._instances)

    def testSkoptBayesianMinOptimizer(self):
        # No need to test correctness since we are relying
        # on external library -- hopefully they are correct :)
        # So we only test whether our preprocessing is correct
        config = get_example_config()
        param_space = config[1]["paramSpace"]
        optimizer = optimizer_ops.SkoptBayesianMinOptimizer(param_space)

        # Manually created the dimensions
        expected_params = [
            skopt_space.Integer(low=1, high=3),
            skopt_space.Integer(low=0, high=3),
            skopt_space.Integer(low=0, high=3),
            skopt_space.Real(low=0, high=3),
            skopt_space.Categorical(categories=("c1", "c2", "c5"))]


        self.assertEqual(expected_params, optimizer._dimensions)


if __name__ == "__main__":
    tf.test.main()
