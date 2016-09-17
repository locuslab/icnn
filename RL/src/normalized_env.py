# Code from Repo SimonRamstedt/ddpg
# Heavily modified

import gym
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('reward_k', 1, 'reward factor')

def make_normalized_env(env):
    """ crate a new environment class with actions and states normalized to [-1,1] """
    act_space = env.action_space
    obs_space = env.observation_space
    if not type(act_space) == gym.spaces.box.Box:
        raise RuntimeError('Environment with continous action space (i.e. Box) required.')
    if not type(obs_space) == gym.spaces.box.Box:
        raise RuntimeError('Environment with continous observation space (i.e. Box) required.')

    env_type = type(env)

    class FilteredEnv(env_type):

        def __init__(self):
            # transfer properties
            self.__dict__.update(env.__dict__)

            # Observation space
            if np.any(obs_space.high < 1e10) and np.any(obs_space.low > -1e10):
                h = obs_space.high
                l = obs_space.low
                sc = h - l
                self.obs_k = sc / 2.
                self.obs_b = (h + l) / 2.
            else:
                self.obs_k = np.ones_like(obs_space.high)
                self.obs_b = np.zeros_like(obs_space.high)

            # Action space
            h = act_space.high
            l = act_space.low
            sc = h - l
            self.act_k = sc / 2.
            self.act_b = (h + l) / 2.

            # Rewards
            self.reward_k = FLAGS.reward_k
            self.reward_b = 0.

            # Check and assign transformed spaces
            self.observation_space = gym.spaces.Box(self.filter_observation(obs_space.low),
                                                    self.filter_observation(obs_space.high))
            self.action_space = gym.spaces.Box(-np.ones_like(act_space.high),
                                               np.ones_like(act_space.high))

            def assertEqual(a, b): assert np.all(a == b), "{} != {}".format(a, b)
            assertEqual(self.filter_action(self.action_space.low), act_space.low)
            assertEqual(self.filter_action(self.action_space.high), act_space.high)

        def filter_observation(self, obs):
            return (obs - self.obs_b) / self.obs_k

        def filter_action(self, action):
            return self.act_k * action + self.act_b

        def filter_reward(self, reward):
            ''' has to be applied manually otherwise it makes the reward_threshold invalid '''
            return self.reward_k * reward + self.reward_b

        def step(self, action):
            ac_f = np.clip(self.filter_action(action), self.action_space.low, self.action_space.high)
            obs, reward, term, info = env_type.step(self, ac_f)  # super function
            obs_f = self.filter_observation(obs)
            return obs_f, reward, term, info

    fenv = FilteredEnv()

    print('True action space: ' + str(act_space.low) + ', ' + str(act_space.high))
    print('True state space: ' + str(obs_space.low) + ', ' + str(obs_space.high))
    print('Filtered action space: ' + str(fenv.action_space.low) + ', ' + str(fenv.action_space.high))
    print('Filtered state space: ' + str(fenv.observation_space.low) + ', ' + str(fenv.observation_space.high))

    return fenv
