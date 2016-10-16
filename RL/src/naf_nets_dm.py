import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

def theta(dimIn, dimOut, l1, l2, scope):
    with tf.variable_scope(scope):
        normal_init = tf.truncated_normal_initializer(mean=0.0, stddev=FLAGS.initstd)
        return [tf.get_variable(name='w1', shape=[dimIn, l1], initializer=normal_init),
                tf.get_variable(name='b1', shape=[l1], initializer=tf.constant_initializer(0.0)),
                tf.get_variable(name='w2', shape=[l1, l2], initializer=normal_init),
                tf.get_variable(name='b2', shape=[l2], initializer=tf.constant_initializer(0.0)),
                tf.get_variable(name='w3', shape=[l2, dimOut], initializer=normal_init),
                tf.get_variable(name='b3', shape=[dimOut], initializer=tf.constant_initializer(0.0))]


def build_NN_two_hidden_layers(x, theta, reuse, is_training):
    batch_norm_params = {'is_training': is_training, 'decay': 0.999, 'epsilon': 1e-3,
                         'updates_collections': None, 'reuse': reuse}
    if FLAGS.naf_bn:
        h0 = tf.contrib.layers.batch_norm(x, scope='h0', **batch_norm_params)
    else:
        h0 = x
    h1 = tf.matmul(h0, theta[0]) + theta[1]
    if FLAGS.naf_bn:
        h1 = tf.contrib.layers.batch_norm(h1, scope='h1', **batch_norm_params)
    h1 = tf.nn.relu(h1)
    h2 = tf.matmul(h1, theta[2]) + theta[3]
    if FLAGS.naf_bn:
        h2 = tf.contrib.layers.batch_norm(h2, scope='h2', **batch_norm_params)
    h2 = tf.nn.relu(h2)
    h3 = tf.matmul(h2, theta[4]) + theta[5]
    return h3


def lfunction(obs, theta, reuse, is_training, scope="lfunction"):
    with tf.variable_scope(scope):
        l = build_NN_two_hidden_layers(obs, theta, reuse, is_training)
        return l


def vec2trimat(vec, dim):
    L = tf.reshape(vec, [-1, dim, dim])
    L = tf.batch_matrix_band_part(L, -1, 0) - tf.batch_matrix_diag(tf.batch_matrix_diag_part(L)) + \
        tf.batch_matrix_diag(tf.exp(tf.batch_matrix_diag_part(L)))
    return L


def ufunction(obs, theta, reuse, is_training, scope="ufunction"):
    with tf.variable_scope(scope):
        act = build_NN_two_hidden_layers(obs, theta, reuse, is_training)
        act = tf.tanh(act)
        return act


def afunction(action, lvalue, uvalue, dimA, scope="afunction"):
    with tf.variable_scope(scope):
        delta = action - uvalue
        L = vec2trimat(lvalue, dimA)

        h1 = tf.reshape(delta, [-1, 1, dimA])
        h1 = tf.batch_matmul(h1, L)  # batch:1:dimA
        h1 = tf.squeeze(h1, [1])  # batch:dimA
        h2 = -tf.constant(0.5) * tf.reduce_sum(h1 * h1, 1)  # batch

        return h2


def qfunction(obs, avalue, theta, reuse, is_training, scope="qfunction"):
    with tf.variable_scope(scope):
        q = build_NN_two_hidden_layers(obs, theta, reuse, is_training)
        q = tf.squeeze(q, [1]) + avalue
        return q
