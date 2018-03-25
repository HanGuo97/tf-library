"""exp3 test"""
from __future__ import division
from __future__ import print_function

import math
import numpy as np
from TFLibrary.utils import test_utils
from TFLibrary.Bandits.exp3 import Exp3S


def calc_probs(weights, epsilon):
    probs = []
    num_actions = len(weights)
    for w in weights:
        p = (1 - epsilon) * w / sum(weights) + epsilon / num_actions
        probs.append(p)
    return probs


def weights_update(reward, chosen, weights, epsilon, alpha):
    new_weights = []
    K = len(weights)
    probs = calc_probs(weights, epsilon=epsilon)
    for i, w in enumerate(weights):
        if i == chosen:
            r = reward / probs[i]
        else:
            r = 0
            
        new_w = (w * np.exp(epsilon * r / K) +
                 np.exp(1) * alpha / K * sum(weights))
        
        new_weights.append(new_w)
    
    return new_weights


def random_integers(high):
    return test_utils.random_integers(
        low=0, high=high, shape=[], dtype=np.int32).tolist()


def test():
    raise NotImplementedError
    
    # set up
    alpha = 0.35
    epsilon = 0.95
    num_actions = 11
    max_reward = 100
    weights = [1.0 for _ in range(num_actions)]
    probs_history = []
    rewards_history = []
    weights_history = []
    actions_history = []

    bandits = Exp3S(
        num_actions=num_actions,
        epsilon=epsilon,
        alpha=alpha,
        initial_weights=1.0)

    for i in range(100):
        bandits.sample()
        probs = calc_probs(
            weights=weights, epsilon=epsilon)
        probs_history.append(probs)

        reward = random_integers(max_reward)
        action = random_integers(num_actions)
        rewards_history.append(reward)
        actions_history.append(action)

        bandits.update(
            reward=reward, chosen_arm=action)
        weights = weights_update(
            reward=reward, chosen=action,
            weights=weights, epsilon=epsilon, alpha=alpha)
        weights_history.append(weights)

    return bandits, probs_history, weights_history
