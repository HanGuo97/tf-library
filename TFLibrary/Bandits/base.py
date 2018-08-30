from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


class BaseBandit(object):
    def __init__(self):
        pass

    def sample(self, step=0):
        raise NotImplementedError

    def update(self, reward, chosen_arm):
        raise NotImplementedError

    @property
    def arm_weights(self):
        raise NotImplementedError

    @property
    def reward_histories(self):
        raise NotImplementedError

    def save(self, file_dir):
        raise NotImplementedError

    def load(self, file_dir):
        raise NotImplementedError
