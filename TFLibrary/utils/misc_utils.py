import os
import random
import pickle
import numpy as np
from collections import deque


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def maybe_delete_file(file_dir, check_exists=False):
    if os.path.exists(file_dir):
        os.remove(file_dir)
        print("File %s is deleted" % file_dir)
    
    elif check_exists:
        raise ValueError("File %s does not exist" % file_dir)


class ReplayBuffer(object):
    """
    Data structure for implementing experience replay

    https://github.com/pemami4911/deep-rl

    Author: Patrick Emami
    """
    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0


# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class OrnsteinUhlenbeckActionNoise(object):
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = (self.x_prev + self.theta *
             (self.mu - self.x_prev) * self.dt +
             self.sigma * np.sqrt(self.dt) *
             np.random.normal(size=self.mu.shape))
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = (self.x0 if self.x0 is not None
                       else np.zeros_like(self.mu))

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)