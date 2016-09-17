# Code from Repo SimonRamstedt/ddpg
# Heavily modified

import numpy as np
import tensorflow as tf


def fanin_init(shape, fanin=None):
    fanin = fanin or shape[0]
    v = 1 / np.sqrt(fanin)
    return tf.random_uniform(shape, minval=-v, maxval=v)


def theta_p(dimO, dimA, l1, l2):
    dimO = dimO[0]
    dimA = dimA[0]
    with tf.variable_scope("theta_p"):
        return [tf.Variable(fanin_init([dimO, l1]), name='1w'),
                tf.Variable(fanin_init([l1], dimO), name='1b'),
                tf.Variable(fanin_init([l1, l2]), name='2w'),
                tf.Variable(fanin_init([l2], l1), name='2b'),
                tf.Variable(tf.random_uniform([l2, dimA], -3e-3, 3e-3), name='3w'),
                tf.Variable(tf.random_uniform([dimA], -3e-3, 3e-3), name='3b')]


def policy(obs, theta, name='policy'):
    with tf.variable_op_scope([obs], name, name):
        h0 = tf.identity(obs, name='h0-obs')
        h1 = tf.nn.relu(tf.matmul(h0, theta[0]) + theta[1], name='h1')
        h2 = tf.nn.relu(tf.matmul(h1, theta[2]) + theta[3], name='h2')
        h3 = tf.identity(tf.matmul(h2, theta[4]) + theta[5], name='h3')
        action = tf.nn.tanh(h3, name='h4-action')
        return action


def theta_q(dimO, dimA, l1, l2):
    dimO = dimO[0]
    dimA = dimA[0]
    with tf.variable_scope("theta_q"):
        return [tf.Variable(fanin_init([dimO, l1]), name='1w'),
                tf.Variable(fanin_init([l1], dimO), name='1b'),
                tf.Variable(fanin_init([l1 + dimA, l2]), name='2w'),
                tf.Variable(fanin_init([l2], l1 + dimA), name='2b'),
                tf.Variable(tf.random_uniform([l2, 1], -3e-4, 3e-4), name='3w'),
                tf.Variable(tf.random_uniform([1], -3e-4, 3e-4), name='3b')]


def qfunction(obs, act, theta, name="qfunction"):
    with tf.variable_op_scope([obs, act], name, name):
        h0 = tf.identity(obs, name='h0-obs')
        h0a = tf.identity(act, name='h0-act')
        h1 = tf.nn.relu(tf.matmul(h0, theta[0]) + theta[1], name='h1')
        h1a = tf.concat(1, [h1, act])
        h2 = tf.nn.relu(tf.matmul(h1a, theta[2]) + theta[3], name='h2')
        qs = tf.matmul(h2, theta[4]) + theta[5]
        q = tf.squeeze(qs, [1], name='h3-q')
        return q
