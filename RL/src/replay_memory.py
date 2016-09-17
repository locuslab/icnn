# Code from Repo SimonRamstedt/ddpg
# Heavily modified

import random

import numpy as np

class ReplayMemory:

    def __init__(self, size, dimO, dimA, dtype=np.float32):
        self.size = size
        so = np.concatenate(np.atleast_1d(size, dimO), axis=0)
        sa = np.concatenate(np.atleast_1d(size, dimA), axis=0)
        self.observations = np.empty(so, dtype=dtype)
        self.actions = np.empty(sa, dtype=np.float32)
        self.rewards = np.empty(size, dtype=np.float32)
        self.terminals = np.empty(size, dtype=np.bool)
        self.info = np.empty(size, dtype=object)

        self.n = 0
        self.i = 0

    def reset(self):
        self.n = 0
        self.i = 0

    def enqueue(self, observation, terminal, action, reward, info=None):
        self.observations[self.i, ...] = observation
        self.terminals[self.i] = terminal
        self.actions[self.i, ...] = action
        self.rewards[self.i] = reward
        self.info[self.i, ...] = info
        self.i = (self.i + 1) % self.size
        self.n = min(self.size - 1, self.n + 1)

    def minibatch(self, size):
        indices = np.zeros(size,dtype=np.int)
        for k in range(size):
            invalid = True
            while invalid:
                # sample index ignore wrapping over buffer
                i = np.random.randint(0, self.n - 1)
                # if i-th sample is current one or is terminal: get new index
                if i != self.i and not self.terminals[i]:
                    invalid = False
            indices[k] = i

        o = self.observations[indices, ...]
        a = self.actions[indices]
        r = self.rewards[indices]
        o2 = self.observations[indices + 1, ...]
        t2 = self.terminals[indices + 1]
        info = self.info[indices, ...]

        return o, a, r, o2, t2, info

    def __repr__(self):
        indices = range(0, self.n)
        o = self.observations[indices, ...]
        a = self.actions[indices]
        r = self.rewards[indices]
        t = self.terminals[indices]
        info = self.info[indices, ...]

        s = """
    OBSERVATIONS
    {}

    ACTIONS
    {}

    REWARDS
    {}

    TERMINALS
    {}
    """.format(o, a, r, t)

        return s
