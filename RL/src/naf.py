import os

import numpy as np
import tensorflow as tf

import naf_nets_dm
from replay_memory import ReplayMemory

flags = tf.app.flags
FLAGS = flags.FLAGS

# Continuous Q learning with Normalized Advantage Function


class Agent:

    def __init__(self, dimO, dimA):
        dimA = list(dimA)
        dimO = list(dimO)

        tau = FLAGS.tau
        discount = FLAGS.discount
        l2norm = FLAGS.l2norm
        learning_rate = FLAGS.rate
        outheta = FLAGS.outheta
        ousigma = FLAGS.ousigma

        nets = naf_nets_dm

        # init replay memory
        self.rm = ReplayMemory(FLAGS.rmsize, dimO, dimA)
        # start tf session
        self.sess = tf.Session(config=tf.ConfigProto(
            inter_op_parallelism_threads=FLAGS.thread,
            log_device_placement=False,
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1)))

        is_training = tf.placeholder(tf.bool)

        # create tf computational graph
        self.theta_L = nets.theta(dimO[0], dimA[0] * dimA[0], FLAGS.l1size, FLAGS.l2size, 'theta_L')
        self.theta_U = nets.theta(dimO[0], dimA[0], FLAGS.l1size, FLAGS.l2size, 'theta_U')
        self.theta_V = nets.theta(dimO[0], 1, FLAGS.l1size, FLAGS.l2size, 'theta_V')
        self.theta_Vt, update_Vt = exponential_moving_averages(self.theta_V, tau)

        obs_single = tf.placeholder(tf.float32, [1] + dimO, "obs-single")
        act_test = nets.ufunction(obs_single, self.theta_U, False, is_training)

        # explore
        noise_init = tf.zeros([1] + dimA)
        noise_var = tf.Variable(noise_init, name="noise", trainable=False)
        self.ou_reset = noise_var.assign(noise_init)
        noise = noise_var.assign_sub((outheta) * noise_var - tf.random_normal(dimA, stddev=ousigma))
        act_expl = act_test + noise

        # training
        obs_train = tf.placeholder(tf.float32, [FLAGS.bsize] + dimO, 'obs_train')
        act_train = tf.placeholder(tf.float32, [FLAGS.bsize] + dimA, "act_train")
        rew = tf.placeholder(tf.float32, [FLAGS.bsize], "rew")
        obs2 = tf.placeholder(tf.float32, [FLAGS.bsize] + dimO, "obs2")
        term2 = tf.placeholder(tf.bool, [FLAGS.bsize], "term2")
        # q
        lmat = nets.lfunction(obs_train, self.theta_L, False, is_training)
        uvalue = nets.ufunction(obs_train, self.theta_U, True, is_training)
        avalue = nets.afunction(act_train, lmat, uvalue, dimA[0])
        q_train = nets.qfunction(obs_train, avalue, self.theta_V, False, is_training)

        # q targets
        q2 = nets.qfunction(obs2, tf.constant([0.] * FLAGS.bsize),
                            self.theta_Vt, True, is_training)
        q_target = tf.stop_gradient(tf.select(term2, rew, rew + discount * q2))

        # q loss
        td_error = q_train - q_target
        ms_td_error = tf.reduce_mean(tf.square(td_error), 0)
        theta = self.theta_L + self.theta_U + self.theta_V
        wd_q = tf.add_n([l2norm * tf.nn.l2_loss(var) for var in theta])  # weight decay
        loss_q = ms_td_error + wd_q
        # q optimization
        optim_q = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-4)
        grads_and_vars_q = optim_q.compute_gradients(loss_q)
        optimize_q = optim_q.apply_gradients(grads_and_vars_q)
        with tf.control_dependencies([optimize_q]):
            train_q = tf.group(update_Vt)

        summary_writer = tf.train.SummaryWriter(os.path.join(FLAGS.outdir, 'board'), self.sess.graph)
        summary_list = []
        summary_list.append(tf.scalar_summary('Qvalue', tf.reduce_mean(q_train)))
        summary_list.append(tf.scalar_summary('loss', ms_td_error))
        summary_list.append(tf.scalar_summary('reward', tf.reduce_mean(rew)))

        # tf functions
        with self.sess.as_default():
            self._act_test = Fun([obs_single, is_training], act_test)
            self._act_expl = Fun([obs_single, is_training], act_expl)
            self._reset = Fun([], self.ou_reset)
            self._train = Fun([obs_train, act_train, rew, obs2, term2, is_training], [train_q, loss_q], summary_list, summary_writer)

        # initialize tf variables
        self.saver = tf.train.Saver(max_to_keep=1)
        ckpt = tf.train.latest_checkpoint(FLAGS.outdir + "/tf")
        if ckpt:
            self.saver.restore(self.sess, ckpt)
        else:
            self.sess.run(tf.initialize_all_variables())

        self.sess.graph.finalize()

        self.t = 0  # global training time (number of observations)

    def reset(self, obs):
        self._reset()
        self.observation = obs  # initial observation

    def act(self, test=False):
        obs = np.expand_dims(self.observation, axis=0)
        action = self._act_test(obs, False) if test else self._act_expl(obs, False)
        action = np.clip(action, -1, 1)
        self.action = np.atleast_1d(np.squeeze(action, axis=0))  # TODO: remove this hack
        return self.action

    def observe(self, rew, term, obs2, test=False):

        obs1 = self.observation
        self.observation = obs2

        # train
        if not test:
            self.t = self.t + 1
            self.rm.enqueue(obs1, term, self.action, rew)

            if self.t > FLAGS.warmup:
                for i in range(FLAGS.iter):
                    loss = self.train()

    def train(self):
        obs, act, rew, ob2, term2, info = self.rm.minibatch(size=FLAGS.bsize)
        _, loss = self._train(obs, act, rew, ob2, term2, True, log=FLAGS.summary, global_step=self.t)
        return loss

    def __del__(self):
        self.sess.close()


# Tensorflow utils
#
class Fun:
    """ Creates a python function that maps between inputs and outputs in the computational graph. """

    def __init__(self, inputs, outputs, summary_ops=None, summary_writer=None, session=None):
        self._inputs = inputs if type(inputs) == list else [inputs]
        self._outputs = outputs
        self._summary_op = tf.merge_summary(summary_ops) if type(summary_ops) == list else summary_ops
        self._session = session or tf.get_default_session()
        self._writer = summary_writer

    def __call__(self, *args, **kwargs):
        """
        Arguments:
          **kwargs: input values
          log: if True write summary_ops to summary_writer
          global_step: global_step for summary_writer
        """
        log = kwargs.get('log', False)

        feeds = {}
        for (argpos, arg) in enumerate(args):
            feeds[self._inputs[argpos]] = arg

        out = self._outputs + [self._summary_op] if log else self._outputs
        res = self._session.run(out, feeds)

        if log:
            i = kwargs['global_step']
            self._writer.add_summary(res[-1], global_step=i)
            res = res[:-1]

        return res


def exponential_moving_averages(theta, tau=0.001):
    ema = tf.train.ExponentialMovingAverage(decay=1 - tau)
    update = ema.apply(theta)  # also creates shadow vars
    averages = [ema.average(x) for x in theta]
    return averages, update
