import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

def lrelu(x, p=0.3):
    return tf.maximum(x, p * x)

def theta(dimO, dimA, l1, l2, scope):
    with tf.variable_scope(scope):
        normal_init = tf.truncated_normal_initializer(mean=0.0, stddev=FLAGS.initstd)
        return [tf.get_variable(name='Wx0', shape=[dimO, l1], initializer=normal_init),
                tf.get_variable(name='bx0', shape=[l1], initializer=tf.constant_initializer(0.0)),
                tf.get_variable(name='Wx1', shape=[l1, l2], initializer=normal_init),
                tf.get_variable(name='bx1', shape=[l2], initializer=tf.constant_initializer(0.0)),

                tf.get_variable(name='Wxz0', shape=[dimO, l1], initializer=normal_init),
                tf.get_variable(name='Wxz1', shape=[l1, l2], initializer=normal_init),
                tf.get_variable(name='Wxz2', shape=[l2, 1], initializer=normal_init),

                tf.get_variable(name='Wz1', shape=[l1, l2], initializer=normal_init),
                tf.get_variable(name='Wz2', shape=[l2, 1], initializer=normal_init),

                tf.get_variable(name='Wy0', shape=[dimA, l1], initializer=normal_init),
                tf.get_variable(name='by0', shape=[l1], initializer=tf.constant_initializer(0.0)),
                tf.get_variable(name='Wy1', shape=[dimA, l2], initializer=normal_init),
                tf.get_variable(name='by1', shape=[l2], initializer=tf.constant_initializer(0.0)),
                tf.get_variable(name='Wy2', shape=[dimA, 1], initializer=normal_init),
                tf.get_variable(name='by2', shape=[1], initializer=tf.constant_initializer(0.0))]


def qfunction(obs, act, theta, name="qfunction"):

    with tf.variable_op_scope([obs, act], name, name):
        x = tf.identity(obs, name='h0-obs')

        y = tf.identity(act, name='h0-act')

        u1 = tf.matmul(x, theta[0]) + theta[1]
        u1 = tf.nn.relu(u1)

        u2 = tf.matmul(u1, theta[2]) + theta[3]
        u2 = tf.nn.relu(u2)

        cz1 = tf.matmul(x, theta[4]) + theta[10]
        z1 = tf.matmul(y, theta[9]) + cz1
        z1 = lrelu(z1, FLAGS.lrelu)

        cz2 = tf.matmul(u1, theta[5]) + theta[12]
        z2 = tf.matmul(y, theta[11]) + tf.matmul(z1, tf.abs(theta[7])) + cz2
        z2 = lrelu(z2, FLAGS.lrelu)

        cz3 = tf.matmul(u2, theta[6]) + theta[14]
        z3 = tf.matmul(y, theta[13]) + tf.matmul(z2, tf.abs(theta[8])) + cz3
        z3 = -tf.squeeze(z3, [1], name='z3')

        return z3, cz1, cz2, cz3, z1, z2, u1, u2
