"""
Exp3S Bandits, implementation based on:
-- https://github.com/johnmyleswhite/BanditsBook/blob/master/python/algorithms/exp3/exp3.py
"""
from __future__ import division
from __future__ import print_function

import os
import copy
import math
import random
import pickle

from TFLibrary.utils import tensorflow_utils as tf_utils


def categorical_draw(probs):
    z = random.random()
    cum_prob = 0.0
    for i in range(len(probs)):
        prob = probs[i]
        cum_prob += prob
        if cum_prob > z:
            return i

    return len(probs) - 1


def indicator_fn(value, expr):
    if expr is True:
        return value
    return 0


class Exp3(object):

    def __init__(self, num_actions, initial_weights, epsilon=0.95):
        tf_utils.untested_warning()

        self._epsilon = epsilon
        self._num_actions = num_actions
        self._weights = [initial_weights
            for _ in range(num_actions)]

        self._histories = []

    def sample(self, step=None):
        # Step is unused, kept for compatability
        probs = self._calc_probs()
        chosen_arm = categorical_draw(probs)
        
        _weights = copy.deepcopy(self._weights)
        self._histories.append([_weights, chosen_arm])

        return chosen_arm, probs
        

    def update(self, reward, chosen_arm):
        epsilon = self._epsilon
        num_actions = self._num_actions
        chosen_weight = self._weights[chosen_arm]

        probs = self._calc_probs()
        scaled_reward = reward / probs[chosen_arm]
        growth_factor = math.exp((epsilon / num_actions) * scaled_reward)
        self._weights[chosen_arm] = chosen_weight * growth_factor


    def _calc_probs(self):
        epsilon = self._epsilon
        num_actions = self._num_actions
        weights = copy.deepcopy(self._weights)
        total_weight = sum(weights)
        probs = [0.0 for i in range(num_actions)]

        for arm in range(num_actions):
            probs[arm] = (1 - epsilon) * (weights[arm] / total_weight)
            probs[arm] = probs[arm] + (epsilon) * (1.0 / float(num_actions))

        return probs

    @property
    def arm_weights(self):
        return copy.deepcopy(self._weights)

    def save(self, file_dir):
        with open(file_dir, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

        print("INFO: Successfully Saved MABSelector to ", file_dir)

    def load(self, file_dir):
        if not os.path.exists(file_dir):
            raise ValueError("File not exist ", file_dir)

        with open(file_dir, "rb") as f:
            dump = pickle.load(f)

        if not self._num_actions == dump._num_actions:
            raise ValueError("num_actions incompatible")

        self._weights = dump._weights
        self._histories = dump._histories

        print("INFO: Successfully Loaded MABSelector from ", file_dir)
