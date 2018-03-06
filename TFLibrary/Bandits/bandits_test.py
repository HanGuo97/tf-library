from __future__ import print_function

import os
import numpy as np
from TFLibrary.Bandits import bandits
from TFLibrary.utils import test_utils


def test():
    num_actions = 9
    selector_Q_initial = -2
    average_controller = bandits.MultiArmedBanditSelector(
        num_actions=num_actions,
        Q_initial=selector_Q_initial,
        update_method="average")
    test_average_update(average_controller)

    gradient_controller = bandits.MultiArmedBanditSelector(
        num_actions=num_actions,
        Q_initial=selector_Q_initial,
        update_method="gradient_bandit", alpha=0.30)
    test_gradient_update(gradient_controller, alpha=0.30)

    gradient_controller = bandits.MultiArmedBanditSelector(
        num_actions=num_actions,
        Q_initial=selector_Q_initial,
        update_method="gradient_bandit", alpha=0.5)
    test_gradient_update(gradient_controller, alpha=0.5)

    gradient_controller = bandits.MultiArmedBanditSelector(
        num_actions=num_actions,
        Q_initial=selector_Q_initial,
        update_method="gradient_bandit", alpha=0.7)
    test_gradient_update(gradient_controller, alpha=0.7)

    a_controller = bandits.MultiArmedBanditSelector(
        num_actions=num_actions,
        Q_initial=selector_Q_initial,
        update_method="gradient_bandit")
    another_controller = bandits.MultiArmedBanditSelector(
        num_actions=num_actions,
        Q_initial=selector_Q_initial,
        update_method="gradient_bandit")
    test_saving_and_loading(a_controller, another_controller)



def test_average_update(controller):
    randints = []
    num_actions = controller._num_actions
    Q_initial = controller._Q_values[0].Value
    for _ in range(100):
        rint = np.random.randint(low=0, high=100, size=[])
        ridx = int(np.random.randint(low=0, high=num_actions, size=[]))
        
        randints.append((rint, ridx))
        controller.update_Q_values(rint, ridx)
    
    for i in range(num_actions):
        expected_value = np.mean(
            [x for x, y in randints if y == i] + [Q_initial])
        test_utils.display("%r     %.5f == %.5f" % (
            abs(controller._Q_values[i].Value - expected_value) < 1e-11,
            controller._Q_values[i].Value, expected_value))


def test_gradient_update(controller, alpha=0.3):
    randints = []
    num_actions = controller._num_actions
    Q_initial = controller._Q_values[0].Value
    for _ in range(100):
        rint = np.random.randint(low=0, high=100, size=[])
        ridx = int(np.random.randint(low=0, high=num_actions, size=[]))
        controller.update_Q_values(rint, ridx)
        randints.append((rint, ridx))
    
    for i in range(num_actions):
        subset = [x for x, y in randints if y == i]
        n = len(subset)
        initial = (1 - alpha) ** n * Q_initial
        later = [alpha * (1 - alpha) ** (n - j - 1) * r
                for j, r in enumerate(subset)]
        
        expected_value = initial + sum(later)
        test_utils.display("%r     %.5f == %.5f" % (
            abs(controller._Q_values[i].Value - expected_value) < 1e-11,
            controller._Q_values[i].Value, expected_value))


def test_saving_and_loading(a_controller, another_controller):
    file_dir = "./controller.pkl"
    if os.path.exists(file_dir):
        os.remove(file_dir)

    num_actions = a_controller._num_actions
    for _ in range(100):
        rint = np.random.randint(low=0, high=100, size=[])
        ridx = int(np.random.randint(low=0, high=num_actions, size=[]))

        a_controller.update_Q_values(rint, ridx)
        a_controller.sample()

    print(a_controller._Q_values != another_controller._Q_values)
    print(a_controller._histories != another_controller._histories)
    
    a_controller.save(file_dir)
    another_controller.load(file_dir)

    print(a_controller._Q_values == another_controller._Q_values)
    print(all([
        (a[0] == b[0]).all() and a[1] == b[1]
        for a, b in zip(
            a_controller._histories,
            another_controller._histories)]))
    os.remove(file_dir)


if __name__ == "__main__":
    test()
