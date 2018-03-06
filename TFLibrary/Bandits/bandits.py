from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pickle
import numpy as np
from namedlist import namedlist

Q_Entry = namedlist("Q_Entry", ("Value", "Count"))


def softmax(X, theta=1.0, axis=None):
    """Compute the softmax of each element along an axis of X.
    https://nolanbconaway.github.io/blog/2017/softmax-numpy
    """
    # make X at least 2d
    y = np.atleast_2d(X)
    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    # multiply y against the theta parameter,
    y = y * float(theta)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    # flatten if X was 1D
    if len(X.shape) == 1:
        p = p.flatten()

    return p


def incremental_weighted_mean(V, W, C, G):
    """
    Page 89, Equation 5.7
    http://incompleteideas.net/book/bookdraft2017nov5.pdf
    
    V: incremental weighted mean from last time step
    W: weight of the new value
    C: count of previous values
    G: new value

    Calculate:
        V = sum_k(W_k * G_k) / sum_k(W_k)
    
    Incremental Version:
        C_n+1 = C_n + W_n
        V_n+1 = V_n + W_n / C_n+1 [G_n - V_n]

    """
    _C = C + W
    _V = V + W / _C * (G - V)
    # _C = C
    # _V = V + W * (G - V)
    return _V, _C


def gradient_bandit(old_Q, reward, alpha):
    new_Q = old_Q + alpha * (reward - old_Q)
    return new_Q


def convert_to_one_hot(action_id, action_space):
    return np.eye(action_space)[action_id]


def boltzmann_exploration(Q_values, temperature=1.0):
    # for numerical stability, add 1e-7
    Q_probs = softmax(Q_values, theta=1 / (temperature + 1e-7))
    action = np.random.choice(len(Q_probs), p=Q_probs)
    return action, Q_probs


class MultiArmedBanditSelector(object):
    def __init__(self,
                 num_actions,
                 Q_initial,
                 update_method="average",
                 alpha=0.3,
                 initial_temperature=1.0,
                 temperature_anneal_rate=None,
                 log_history=True):
        if update_method not in ["average", "gradient_bandit"]:
            raise ValueError("Unknown update_method ", update_method)

        self._Q_values = [
            Q_Entry(Value=Q_initial, Count=1)
            for _ in range(num_actions)]
        self._num_actions = num_actions
        self._update_method = update_method

        self._alpha = alpha
        self._temperature = initial_temperature
        self._temperature_anneal_rate = temperature_anneal_rate
        
        # save past selections for debugging
        self._histories = []
        self._log_history = log_history


    def sample(self, step=0, one_hot=False):
        """
        If return probs, return the selection
        distributions otherwise, return the sampled actions
        """
        temperature_coef = (
            np.power(self._temperature_anneal_rate, step)
            if self._temperature_anneal_rate is not None else 1)

        chosen_action, Q_probs = boltzmann_exploration(
            Q_values=np.asarray(self.expected_Q_values),
            temperature=self._temperature * temperature_coef)

        if self._log_history:
            self._histories.append([Q_probs, chosen_action])

        if one_hot:
            chosen_action = convert_to_one_hot(
                action_id=chosen_action,
                action_space=self._num_actions)

        return chosen_action, Q_probs

    def update_Q_values(self, new_Q_value, index):
        # uses sampling, set weights = 1
        if not isinstance(index, int):
            raise ValueError("index must be integers")
        if not index < self._num_actions:
            raise ValueError("index out of range")

        if self._update_method == "average":
            new_Q, new_C = incremental_weighted_mean(
                G=new_Q_value, W=1,
                V=self._Q_values[index].Value,
                C=self._Q_values[index].Count)
            self._Q_values[index].Value = new_Q
            self._Q_values[index].Count = new_C

        elif self._update_method == "gradient_bandit":
            new_Q = gradient_bandit(
                reward=new_Q_value,
                alpha=self._alpha,
                old_Q=self._Q_values[index].Value)
            self._Q_values[index].Value = new_Q
            self._Q_values[index].Count += 1

    @property
    def expected_Q_values(self):
        return [Q.Value for Q in self._Q_values]

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
        
        self._Q_values = dump._Q_values
        self._histories = dump._histories

        print("INFO: Successfully Loaded MABSelector from ", file_dir)
