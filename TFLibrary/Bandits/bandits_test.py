from __future__ import division
from __future__ import print_function


import os
import numpy as np
from TFLibrary.Bandits import bandits
from TFLibrary.utils import test_utils


def test():
    num_actions = 9
    initial_weight = -2

    # Average Updates ####################################
    average_controller = bandits.MultiArmedBanditSelector(
        num_actions=num_actions,
        initial_weight=initial_weight,
        update_rate_fn=lambda step: 1. / float(step + 1),
        reward_shaping_fn=lambda reward, histories: reward)
    test_average_update(average_controller)

    # Gradient Updates ####################################
    gradient_controller = bandits.MultiArmedBanditSelector(
        num_actions=num_actions,
        initial_weight=initial_weight,
        update_rate_fn=lambda step: 0.3,
        reward_shaping_fn=lambda reward, histories: reward)
    test_gradient_update(gradient_controller, alpha=0.3)

    gradient_controller = bandits.MultiArmedBanditSelector(
        num_actions=num_actions,
        initial_weight=initial_weight,
        update_rate_fn=lambda step: 0.5,
        reward_shaping_fn=lambda reward, histories: reward)
    test_gradient_update(gradient_controller, alpha=0.5)

    gradient_controller = bandits.MultiArmedBanditSelector(
        num_actions=num_actions,
        initial_weight=initial_weight,
        update_rate_fn=lambda step: 0.7,
        reward_shaping_fn=lambda reward, histories: reward)
    test_gradient_update(gradient_controller, alpha=0.7)


    # Arbitrary Updates ####################################
    gradient_controller = bandits.MultiArmedBanditSelector(
        num_actions=num_actions,
        initial_weight=initial_weight,
        update_rate_fn=lambda step: np.sin(step),
        reward_shaping_fn=lambda reward, histories: reward)
    test_arbitrary_update(gradient_controller)

    gradient_controller = bandits.MultiArmedBanditSelector(
        num_actions=num_actions,
        initial_weight=initial_weight,
        update_rate_fn=lambda step: np.cos(step),
        reward_shaping_fn=lambda reward, histories: reward)
    test_arbitrary_update(gradient_controller)

    # use slightly higher tolerance because
    # test setting `np.sqrt` led to very large
    # numbers, and inprecision can occur
    gradient_controller = bandits.MultiArmedBanditSelector(
        num_actions=num_actions,
        initial_weight=initial_weight,
        update_rate_fn=lambda step: np.sqrt(step) + step / 17.3,
        reward_shaping_fn=lambda reward, histories: reward)
    test_arbitrary_update(gradient_controller, tolerance=1e-5)




    # Save and Restore ####################################
    a_controller = bandits.MultiArmedBanditSelector(
        num_actions=num_actions,
        initial_weight=initial_weight,
        update_rate_fn=lambda step: 0.3,
        reward_shaping_fn=lambda reward, histories: reward / 2.)
    another_controller = bandits.MultiArmedBanditSelector(
        num_actions=num_actions,
        initial_weight=initial_weight,
        update_rate_fn=lambda step: 0.7,
        reward_shaping_fn=lambda reward, histories: reward - 1)
    test_saving_and_loading(a_controller, another_controller)



def test_average_update(controller):
    randints = []
    num_actions = controller._num_actions
    Q_initial = controller._Q_entries[0].Value
    for _ in range(100):
        rint = np.random.randint(low=0, high=100, size=[])
        ridx = int(np.random.randint(low=0, high=num_actions, size=[]))
        
        randints.append((rint, ridx))
        controller.update(rint, ridx)
    
    for i in range(num_actions):
        expected_value = np.mean(
            [x for x, y in randints if y == i] + [Q_initial])
        test_utils.display("%r     %.5f == %.5f" % (
            abs(controller._Q_entries[i].Value - expected_value) < 1e-11,
            controller._Q_entries[i].Value, expected_value))


def test_gradient_update(controller, alpha=0.3):
    randints = []
    num_actions = controller._num_actions
    Q_initial = controller._Q_entries[0].Value
    for _ in range(100):
        rint = np.random.randint(low=0, high=100, size=[])
        ridx = int(np.random.randint(low=0, high=num_actions, size=[]))
        controller.update(rint, ridx)
        randints.append((rint, ridx))
    
    for i in range(num_actions):
        subset = [x for x, y in randints if y == i]
        n = len(subset)
        initial = (1 - alpha) ** n * Q_initial
        later = [alpha * (1 - alpha) ** (n - j - 1) * r
                for j, r in enumerate(subset)]
        
        expected_value = initial + sum(later)
        test_utils.display("%r     %.5f == %.5f" % (
            abs(controller._Q_entries[i].Value - expected_value) < 1e-11,
            controller._Q_entries[i].Value, expected_value))


def _compute_coefs_for_fn(fn, randints, arm_index, Q_initial):
    """
    V' = V + g x (r - V)
       = V + g x r - g x V
       = (1 - g) V + g x r
    """
    filtered_rewards = np.array(
        [x for x, y in randints if y == arm_index])
    shaped_rewards = Q_initial
    for s, r in enumerate(filtered_rewards):
        shaped_rewards = (
            (1 - fn(s + 1)) * shaped_rewards + fn(s + 1) * r)
    
    return shaped_rewards


def test_arbitrary_update(controller, tolerance=1e-11):
    randints = []
    num_actions = controller._num_actions
    Q_initial = controller._Q_entries[0].Value
    for _ in range(100):
        rint = np.random.randint(low=0, high=100, size=[])
        ridx = int(np.random.randint(low=0, high=num_actions, size=[]))

        randints.append((rint, ridx))
        controller.update(rint, ridx)

    for i in range(num_actions):
        expected_value = _compute_coefs_for_fn(
            fn=controller._update_rate_fn,
            randints=randints,
            arm_index=i,
            Q_initial=Q_initial)

        test_utils.display("%r     %.5f == %.5f" % (
            abs(controller._Q_entries[i].Value - expected_value) < tolerance,
            controller._Q_entries[i].Value, expected_value))


def test_saving_and_loading(a_controller, another_controller):
    file_dir = "./controller.pkl"
    if os.path.exists(file_dir):
        os.remove(file_dir)

    num_actions = a_controller._num_actions
    for _ in range(100):
        rint = np.random.randint(low=0, high=100, size=[])
        ridx = int(np.random.randint(low=0, high=num_actions, size=[]))

        a_controller.update(rint, ridx)
        a_controller.sample()

    print(a_controller._Q_entries !=
          another_controller._Q_entries)
    print(a_controller._sample_histories !=
          another_controller._sample_histories)
    print(a_controller._update_histories !=
          another_controller._update_histories)
    
    a_controller.save(file_dir)
    another_controller.load(file_dir)

    print(a_controller._Q_entries == another_controller._Q_entries)
    print(all([
        (a[0] == b[0]).all() and a[1] == b[1]
        for a, b in zip(
            a_controller._sample_histories,
            another_controller._sample_histories)]))
    print(all([
        (a[0] == b[0]).all() and a[1] == b[1] and a[2] == b[2]
        for a, b in zip(
            a_controller._update_histories,
            another_controller._update_histories)]))
    
    for suffix in ["._Q_entries",
                   "._sample_histories",
                   "._update_histories"]:
        os.remove(file_dir + suffix)


if __name__ == "__main__":
    test()
