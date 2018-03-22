from __future__ import division
import copy
import numpy as np


def exp3_update(sampled, rewards, probs, weights, gamma):
    """
    https://cseweb.ucsd.edu/~yfreund/papers/bandits.pdf
    Page 6
    """
    if not len(rewards) == len(probs):
        raise ValueError("rewards and probs have different shapes")
    if not len(rewards) == len(weights):
        raise ValueError("rewards and weights have different shapes")
    if rewards < 0 or rewards > 1:
        raise ValueError("rewards must be in [0,1]")

    K = len(rewards)
    new_weights = copy.deepcopy(weights)
    for j, r in enumerate(rewards):
        if j != sampled:
            r_hat = 0.
        else:
            r_hat = r / probs[j]

        new_weights[j] = weights[j] * np.exp(gamma * r_hat / K)

    probs = exp3_distribution(new_weights, gamma)

    return probs, new_weights


def exp3_distribution(weights, gamma):
    K = len(weights)
    Z = np.sum(weights)
    probabilities = []
    for w in weights:
        p = (1 - gamma) * w / Z + gamma / K
        probabilities.append(p)

    return probabilities
