from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

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


def moving_mean(old_mean, new_value, weight):
    new_mean = old_mean + weight * (new_value - old_mean)
    return new_mean


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
                 initial_temperature=1.0,
                 temperature_anneal_rate=None,
                 log_history=True):
        
        self._Q_values = [
            Q_Entry(Value=Q_initial, Count=1)
            for _ in range(num_actions)]
        self._num_actions = num_actions
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
            if self._temperature_anneal_rate else 1)

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

        new_Q, new_C = incremental_weighted_mean(
            G=new_Q_value, W=1,
            V=self._Q_values[index].Value,
            C=self._Q_values[index].Count)
        self._Q_values[index].Value = new_Q
        self._Q_values[index].Count = new_C


    @property
    def expected_Q_values(self):
        return [Q.Value for Q in self._Q_values]
