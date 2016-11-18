import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

def theta(dimO, dimA, l1, l2, scope):
    with tf.variable_scope(scope):
        normal_init = tf.truncated_normal_initializer(mean=0.0, stddev=FLAGS.initstd)
        return [tf.get_variable(name='Wx0', shape=[dimO, l1], initializer=normal_init),
                tf.get_variable(name='Wx1', shape=[l1, l2], initializer=normal_init),
                tf.get_variable(name='bx0', shape=[l1], initializer=tf.constant_initializer(0.0)),
                tf.get_variable(name='bx1', shape=[l2], initializer=tf.constant_initializer(0.0)),
                # 4
                tf.get_variable(name='Wzu1', shape=[l1, l1], initializer=normal_init),
                tf.get_variable(name='Wzu2', shape=[l2, l2], initializer=normal_init),
                tf.get_variable(name='Wz1', shape=[l1, l2], initializer=normal_init),
                tf.get_variable(name='Wz2', shape=[l2, 1], initializer=normal_init),
                tf.get_variable(name='bz1', shape=[l1], initializer=tf.constant_initializer(1.0)),
                tf.get_variable(name='bz2', shape=[l2], initializer=tf.constant_initializer(1.0)),
                # 10
                tf.get_variable(name='Wyu0', shape=[dimO, dimA], initializer=normal_init),
                tf.get_variable(name='Wyu1', shape=[l1, dimA], initializer=normal_init),
                tf.get_variable(name='Wyu2', shape=[l2, dimA], initializer=normal_init),
                tf.get_variable(name='Wy0', shape=[dimA, l1], initializer=normal_init),
                tf.get_variable(name='Wy1', shape=[dimA, l2], initializer=normal_init),
                tf.get_variable(name='Wy2', shape=[dimA, 1], initializer=normal_init),
                tf.get_variable(name='by0', shape=[dimA], initializer=tf.constant_initializer(1.0)),
                tf.get_variable(name='by1', shape=[dimA], initializer=tf.constant_initializer(1.0)),
                tf.get_variable(name='by2', shape=[dimA], initializer=tf.constant_initializer(1.0)),
                # 19
                tf.get_variable(name='Wu0', shape=[dimO, l1], initializer=normal_init),
                tf.get_variable(name='Wu1', shape=[l1, l2], initializer=normal_init),
                tf.get_variable(name='Wu2', shape=[l2, 1], initializer=normal_init),
                # 22
                tf.get_variable(name='b0', shape=[l1], initializer=tf.constant_initializer(0.0)),
                tf.get_variable(name='b1', shape=[l2], initializer=tf.constant_initializer(0.0)),
                tf.get_variable(name='b2', shape=[1], initializer=tf.constant_initializer(0.0)),
                ]


def qfunction(obs, act, theta, reuse, is_training, name="qfunction"):
    batch_norm_params = {'is_training': is_training, 'decay': 0.999, 'epsilon': 1e-3,
                         'updates_collections': None, 'reuse': reuse}

    with tf.variable_op_scope([obs, act], name, name):
        u0 = tf.identity(obs)
        y = tf.identity(act)

        u1 = tf.matmul(u0, theta[0]) + theta[2]
        u1 = tf.nn.relu(u1)
        u2 = tf.matmul(u1, theta[1]) + theta[3]
        if FLAGS.icnn_bn:
            u2 = tf.nn.relu(tf.contrib.layers.batch_norm(u2, scope='u2', **batch_norm_params))
        else:
            u2 = tf.nn.relu(u2)

        z1 = tf.matmul((tf.matmul(u0, theta[10]) + theta[16]) * y, theta[13])
        z1 = z1 + tf.matmul(u0, theta[19]) + theta[22]
        z1 = tf.nn.relu(z1)

        z2 = tf.matmul(tf.nn.relu(tf.matmul(u1, theta[4]) + theta[8]) * z1, tf.abs(theta[6]))
        z2 = z2 + tf.matmul((tf.matmul(u1, theta[11]) + theta[17]) * y, theta[14])
        z2 = z2 + tf.matmul(u1, theta[20]) + theta[23]
        if FLAGS.icnn_bn:
            z2 = tf.nn.relu(tf.contrib.layers.batch_norm(z2, scope='z2', **batch_norm_params))
        else:
            z2 = tf.nn.relu(z2)

        z3 = tf.matmul(tf.nn.relu(tf.matmul(u2, theta[5]) + theta[9]) * z2, tf.abs(theta[7]))
        z3 = z3 + tf.matmul((tf.matmul(u2, theta[12]) + theta[18]) * y, theta[15])
        z3 = z3 + tf.matmul(u2, theta[21]) + theta[24]
        z3 = -tf.squeeze(z3, [1], name='z3')

        return z3, z1, z2, u1, u2
