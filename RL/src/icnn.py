import os

import numpy as np
import tensorflow as tf

import bundle_entropy
import icnn_nets_dm
from replay_memory import ReplayMemory

flags = tf.app.flags
FLAGS = flags.FLAGS

# Input Convex Neural Network


class Agent:

    def __init__(self, dimO, dimA):
        dimA = list(dimA)
        dimO = list(dimO)
        self.dimA = dimA[0]
        self.dimO = dimO[0]

        tau = FLAGS.tau
        discount = FLAGS.discount
        l2norm = FLAGS.l2norm
        learning_rate = FLAGS.rate
        outheta = FLAGS.outheta
        ousigma = FLAGS.ousigma

        nets = icnn_nets_dm

        def entropy(x): #the real concave entropy function
            x_move_reg = tf.clip_by_value((x + 1) / 2, 0.0001, 0.9999)
            pen = x_move_reg * tf.log(x_move_reg) + (1 - x_move_reg) * tf.log(1 - x_move_reg)
            return -tf.reduce_sum(pen, 1)

        # init replay memory
        self.rm = ReplayMemory(FLAGS.rmsize, dimO, dimA)
        # start tf session
        self.sess = tf.Session(config=tf.ConfigProto(
            inter_op_parallelism_threads=FLAGS.thread,
            log_device_placement=False,
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1)))

        # create tf computational graph
        self.theta = nets.theta(dimO[0], dimA[0], FLAGS.l1size, FLAGS.l2size, 'theta')
        self.theta_t, update_t = exponential_moving_averages(self.theta, tau)

        obs = tf.placeholder(tf.float32, [1] + dimO, "obs")
        act_test = tf.placeholder(tf.float32, [1] + dimA, "act")

        # explore
        noise_init = tf.zeros([1] + dimA)
        noise_var = tf.Variable(noise_init, name="noise", trainable=False)
        self.ou_reset = noise_var.assign(noise_init)
        noise = noise_var.assign_sub((outheta) * noise_var - tf.random_normal(dimA, stddev=ousigma))
        act_expl = act_test + noise

        # test, single sample q function & gradient for bundle method
        q_test_opt, _, _, _, _ = nets.qfunction(obs, act_test, self.theta, False, False)
        loss_test = -q_test_opt
        act_test_grad = tf.gradients(loss_test, act_test)[0]

        loss_test_gd = -q_test_opt - entropy(act_test)
        act_test_grad_gd = tf.gradients(loss_test_gd, act_test)[0]

        # batched q function & gradient for bundle method
        obs_train2_opt = tf.placeholder(tf.float32, [FLAGS.bsize] + dimO, "obs_train2_opt")
        act_train2_opt = tf.placeholder(tf.float32, [FLAGS.bsize] + dimA, "act_train2_opt")

        q_train2_opt, _, _, _, _ = nets.qfunction(obs_train2_opt, act_train2_opt, self.theta_t, True, False)
        loss_train2 = -q_train2_opt
        act_train2_grad = tf.gradients(loss_train2, act_train2_opt)[0]

        loss_train2_gd = -q_train2_opt - entropy(act_train2_opt)
        act_train2_grad_gd = tf.gradients(loss_train2_gd, act_train2_opt)[0]

        # training
        obs_train = tf.placeholder(tf.float32, [FLAGS.bsize] + dimO, "obs_train")
        act_train = tf.placeholder(tf.float32, [FLAGS.bsize] + dimA, "act_train")
        rew = tf.placeholder(tf.float32, [FLAGS.bsize], "rew")
        obs_train2 = tf.placeholder(tf.float32, [FLAGS.bsize] + dimO, "obs_train2")
        act_train2 = tf.placeholder(tf.float32, [FLAGS.bsize] + dimA, "act_train2")
        term2 = tf.placeholder(tf.bool, [FLAGS.bsize], "term2")

        q_train, q_train_z1, q_train_z2, q_train_u1, q_train_u2 = nets.qfunction(obs_train, act_train, self.theta, True, True)
        q_train_entropy = q_train + entropy(act_train)

        q_train2, _, _, _, _ = nets.qfunction(obs_train2, act_train2, self.theta_t, True, True)
        q_train2_entropy = q_train2 + entropy(act_train2)
        q_target = tf.select(term2, rew, rew + discount * q_train2_entropy)
        q_target = tf.maximum(q_train_entropy - 1., q_target)
        q_target = tf.minimum(q_train_entropy + 1., q_target)
        q_target = tf.stop_gradient(q_target)

        # q loss
        td_error = q_train_entropy - q_target
        ms_td_error = tf.reduce_mean(tf.square(td_error), 0)
        theta = self.theta
        wd_q = tf.add_n([l2norm * tf.nn.l2_loss(var) if var.name[6] == 'W' else 0. for var in theta])  # weight decay
        loss_q = ms_td_error + wd_q
        # q optimization
        optim_q = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-4)
        grads_and_vars_q = optim_q.compute_gradients(loss_q)
        optimize_q = optim_q.apply_gradients(grads_and_vars_q)
        with tf.control_dependencies([optimize_q]):
            train_q = tf.group(update_t)


        summary_writer = tf.train.SummaryWriter(os.path.join(FLAGS.outdir, 'board'), self.sess.graph)
        summary_list = []
        summary_list.append(tf.scalar_summary('Qvalue', tf.reduce_mean(q_train_entropy)))
        summary_list.append(tf.scalar_summary('loss', ms_td_error))
        summary_list.append(tf.scalar_summary('reward', tf.reduce_mean(rew)))
        summary_list.append(tf.scalar_summary('cvx_z1', tf.reduce_mean(q_train_z1)))
        summary_list.append(tf.scalar_summary('cvx_z2', tf.reduce_mean(q_train_z2)))
        summary_list.append(tf.scalar_summary('cvx_z1_pos', tf.reduce_mean(tf.to_float(q_train_z1 > 1e-15))))
        summary_list.append(tf.scalar_summary('cvx_z2_pos', tf.reduce_mean(tf.to_float(q_train_z2 > 1e-15))))
        summary_list.append(tf.scalar_summary('noncvx_u1', tf.reduce_mean(q_train_u1)))
        summary_list.append(tf.scalar_summary('noncvx_u2', tf.reduce_mean(q_train_u2)))
        summary_list.append(tf.scalar_summary('noncvx_u1_pos', tf.reduce_mean(tf.to_float(q_train_u1 > 1e-15))))
        summary_list.append(tf.scalar_summary('noncvx_u2_pos', tf.reduce_mean(tf.to_float(q_train_u2 > 1e-15))))

        # tf functions
        with self.sess.as_default():
            self._reset = Fun([], self.ou_reset)
            self._act_expl = Fun(act_test, act_expl)
            self._train = Fun([obs_train, act_train, rew, obs_train2, act_train2, term2], [train_q, loss_q], summary_list, summary_writer)

            self._opt_test = Fun([obs, act_test], [loss_test, act_test_grad])
            self._opt_train = Fun([obs_train2_opt, act_train2_opt], [loss_train2, act_train2_grad])

            self._opt_test_gd = Fun([obs, act_test], [loss_test_gd, act_test_grad_gd])
            self._opt_train_gd = Fun([obs_train2_opt, act_train2_opt], [loss_train2_gd, act_train2_grad_gd])

        # initialize tf variables
        self.saver = tf.train.Saver(max_to_keep=1)
        ckpt = tf.train.latest_checkpoint(FLAGS.outdir + "/tf")
        if ckpt:
            self.saver.restore(self.sess, ckpt)
        else:
            self.sess.run(tf.initialize_all_variables())

        self.sess.graph.finalize()

        self.t = 0  # global training time (number of observations)

    def get_cvx_opt(self, func, obs):
        act = np.ones((obs.shape[0], self.dimA)) * 0.5
        def fg(x):
            value, grad = func(obs, 2 * x - 1)
            grad *= 2
            return value, grad

        act = bundle_entropy.solveBatch(fg, act)[0]
        act = 2 * act - 1

        return act

    def get_cvx_opt_gd(self, func, obs):
        b1 = 0.9
        b2 = 0.999
        lam = 0.7
        eps = 1e-8
        alpha = 0.01
        nBatch = obs.shape[0]
        act = np.zeros((nBatch, self.dimA))
        m = np.zeros_like(act)
        v = np.zeros_like(act)

        b1t, b2t = 1., 1.
        act_best, a_diff, f_best = [None]*3
        for i in range(10000):
            f, g = func(obs, act)

            if i == 0:
                act_best = act.copy()
                f_best = f.copy()
            else:
                I = (f < f_best)
                act_best[I] = act[I]
                f_best[I] = f[I]

            m = b1 * m + (1. - b1) * g
            v = b2 * v + (1. - b2) * (g * g)
            b1t *= b1
            b2t *= b2
            mhat = m/(1.-b1t)
            vhat = v/(1.-b2t)

            prev_act = act.copy()
            act -= alpha * mhat / (np.sqrt(v) + eps)
            act = np.clip(act, -1, 1)

            a_diff_i = np.linalg.norm(act - prev_act)
            a_diff = a_diff_i if a_diff is None else lam*a_diff + (1.-lam)*a_diff_i
            if a_diff < 1e-4:
                print('  + ADAM took {} iterations'.format(i))
                return act_best

        print('  + Warning: ADAM did not converge.')
        return act_best

    def reset(self, obs):
        self._reset()
        self.observation = obs  # initial observation

    def act(self, test=False):
        obs = np.expand_dims(self.observation, axis=0)
        if FLAGS.use_gd:
            act = self.get_cvx_opt_gd(self._opt_test_gd, obs)
        else:
            act = self.get_cvx_opt(self._opt_test, obs)
        action = act if test else self._act_expl(act)
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
                for i in xrange(FLAGS.iter):
                    loss = self.train()

    def train(self):
        obs, act, rew, ob2, term2, info = self.rm.minibatch(size=FLAGS.bsize)
        if FLAGS.use_gd:
            act2 = self.get_cvx_opt_gd(self._opt_train_gd, ob2)
        else:
            act2 = self.get_cvx_opt(self._opt_train, ob2)

        _, loss = self._train(obs, act, rew, ob2, act2, term2, log=FLAGS.summary, global_step=self.t)
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
            res = res[: -1]

        return res


def exponential_moving_averages(theta, tau=0.001):
    ema = tf.train.ExponentialMovingAverage(decay=1 - tau)
    update = ema.apply(theta)  # also creates shadow vars
    averages = [ema.average(x) for x in theta]
    return averages, update
