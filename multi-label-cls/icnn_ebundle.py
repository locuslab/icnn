#!/usr/bin/env python3.4

import tensorflow as tf
import tflearn

import numpy as np
import numpy.random as npr

np.set_printoptions(precision=2)
# np.seterr(all='raise')
np.seterr(all='warn')

import argparse
import csv
import os
import sys
import time
import pickle as pkl
import json
import shutil
import re

import setproctitle

from datetime import datetime

sys.path.append('../lib')

import bibsonomy
import bundle_entropy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='work/icnn.ebundle')
    parser.add_argument('--nEpoch', type=float, default=50)
    parser.add_argument('--trainBatchSz', type=int, default=128)
    # parser.add_argument('--testBatchSz', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--layerSizes', type=int, nargs='+', default=[600, 600])
    parser.add_argument('--noncvx', action='store_true')
    parser.add_argument('--dataset', type=str, choices=['bibtex', 'bookmarks', 'delicious'],
                        default='bibtex')
    parser.add_argument('--valSplit', type=float, default=0)

    args = parser.parse_args()

    setproctitle.setproctitle('bamos.icnn.ebundle.{}.{}'.format(
        args.dataset,
        ','.join(str(x) for x in args.layerSizes)))

    npr.seed(args.seed)
    tf.set_random_seed(args.seed)

    save = os.path.expanduser(args.save)
    if os.path.isdir(save):
        shutil.rmtree(save)
    os.makedirs(save)

    if args.dataset == 'bibtex':
        data = bibsonomy.loadBibtex("data/bibtex")
    elif args.dataset == 'bookmarks':
        data = bibsonomy.loadBookmarks("data/bookmarks")
    elif args.dataset == 'delicious':
        data = bibsonomy.loadDelicious("data/delicious")
    else:
        assert(False)

    # with open('work-ff/best.logits.pkl', 'rb') as f:
        # data = pkl.load(f)

    nTest = data['testX'].shape[0]
    nFeatures = data['trainX'].shape[1]
    nLabels = data['trainY'].shape[1]
    nXy = nFeatures + nLabels

    nTrain_orig = data['trainX'].shape[0]
    nVal = int(args.valSplit*nTrain_orig)
    nTrain = nTrain_orig-nVal
    if args.valSplit > 0:
        I = npr.permutation(nTrain_orig)
        trainI, valI = I[:nTrain], I[nVal:]
        trainX = data['trainX'][trainI, :]
        trainY = data['trainY'][trainI, :]
        valX = data['trainX'][valI, :]
        valY = data['trainY'][valI, :]
    else:
        trainX = data['trainX']
        trainY = data['trainY']

    print("\n\n" + "="*40)
    print("+ nTrain: {}, nTest: {}".format(nTrain, nTest))
    print("+ nFeatures: {}, nLabels: {}".format(nFeatures, nLabels))
    print("="*40 + "\n\n")

    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = Model(nFeatures, nLabels, args.layerSizes, sess)
        if args.valSplit > 0:
            model.train(args, trainX, trainY, valX, valY)
        else:
            model.train(args, trainX, trainY, data['testX'], data['testY'])


def variable_summaries(var, name=None):
    if name is None:
        name = var.name
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stdev'):
            stdev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.scalar_summary('stdev/' + name, stdev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)

def getW(inSz, outSz, name, norm=True):
    with tf.variable_scope("W"):
        stdev = 1./np.sqrt(outSz)
        v = tf.get_variable(name, (inSz, outSz),
                            initializer=tf.random_uniform_initializer(-stdev, stdev))
        variable_summaries(v)
        if norm:
            v_sqnorm = tf.reduce_sum(tf.square(v), 0)
            g = tf.get_variable(name+'.g', (outSz), initializer=tf.constant_initializer(1.0))
            return tf.matmul(v, tf.diag(tf.div(g, v_sqnorm)))
        else:
            return v

def getW_pos(inSz, outSz, name, norm=True):
    with tf.variable_scope("W_pos"):
        stdev = 1./np.sqrt(outSz)
        v = tf.get_variable(name, (inSz, outSz),
                            initializer=tf.random_uniform_initializer(0, stdev))
        variable_summaries(v)
        if norm:
            v_sqnorm = tf.reduce_sum(tf.square(v), 0)
            g = tf.get_variable(name+'.g', (outSz), initializer=tf.constant_initializer(1.0))
            return tf.matmul(v, tf.diag(tf.div(g, v_sqnorm)))
        else:
            return v

def getB(outSz, name='b'):
    v = tf.get_variable(name, outSz, initializer=tf.constant_initializer(0.0))
    variable_summaries(v)
    return v

class Model:
    def __init__(self, nFeatures, nLabels, layerSzs, sess):
        self.nFeatures = nFeatures
        self.nLabels = nLabels
        self.nXy = nFeatures + nLabels
        self.sess = sess

        self.trueY_ = tf.placeholder(tf.float32, shape=(None, nLabels), name='trueY')

        self.x_ = tf.placeholder(tf.float32, shape=(None, nFeatures), name='x')
        self.y_ = tf.placeholder(tf.float32, shape=(None, nLabels), name='y')
        self.v_ = tf.placeholder(tf.float32, shape=(None, nLabels), name='v')
        self.c_ = tf.placeholder(tf.float32, shape=(None), name='c')

        self.E_ = self.f(self.x_, self.y_, layerSzs)
        variable_summaries(self.E_)
        self.E_entr_ = self.E_ + tf.reduce_sum(self.y_*tf.log(self.y_), 1) + \
                       tf.reduce_sum((1.-self.y_)*tf.log(1.-self.y_), 1)
        self.dE_entr_dy_ = tf.gradients(self.E_entr_, self.y_)[0]

        self.theta_ = tf.trainable_variables()
        self.theta_cvx_ = [v for v in self.theta_
                           if 'proj' in v.name and 'W:' in v.name]

        self.makeCvx = [v.assign(tf.abs(v)) for v in self.theta_cvx_]
        self.proj = [v.assign(tf.maximum(v, 0)) for v in self.theta_cvx_]

        self.dE_dy_ = tf.gradients(self.E_, self.y_)[0]

        self.F_ = tf.mul(self.c_, self.E_) + tf.reduce_sum(tf.mul(self.dE_dy_, self.v_), 1)

        # regLosses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # self.F_reg_ = self.F_ + 0.1*sum(regLosses)

        self.opt = tf.train.AdamOptimizer(0.001)
        self.gv_ = [(g,v) for g,v in self.opt.compute_gradients(self.F_, self.theta_)
                    if g is not None]
        self.train_step = self.opt.apply_gradients(self.gv_)

        for g,v in self.gv_:
            variable_summaries(g, 'gradients/'+v.name)

        self.yBinary_ = tf.select(self.y_ < 0.5, tf.zeros_like(self.y_),
                                  tf.ones_like(self.y_))
        self.F1_ = tf.reduce_mean(2*tf_nOnes(tf.mul(self.trueY_, self.yBinary_))/
                                  (tf_nOnes(self.trueY_) + tf_nOnes(self.yBinary_)))

        self.l_yN_ = tf.placeholder(tf.float32, name='l_yN')
        tf.scalar_summary('crossEntr', self.l_yN_)

        self.nBundleIter_ = tf.placeholder(tf.float32, [None], name='nBundleIter')
        variable_summaries(self.nBundleIter_)

        self.nActive_ = tf.placeholder(tf.float32, [None], name='nActive')
        variable_summaries(self.nActive_)

        self.merged = tf.merge_all_summaries()
        self.saver = tf.train.Saver(max_to_keep=1)

    def train(self, args, trainX, trainY, valX, valY):
        save = args.save

        nTrain = trainX.shape[0]
        nTest = valX.shape[0]

        nIter = int(np.ceil(args.nEpoch*nTrain/args.trainBatchSz))

        trainFields = ['iter', 'f1', 'loss']
        trainF = open(os.path.join(save, 'train.csv'), 'w')
        trainW = csv.writer(trainF)
        trainW.writerow(trainFields)

        testFields = ['iter', 'f1', 'loss']
        testF = open(os.path.join(save, 'test.csv'), 'w')
        testW = csv.writer(testF)
        testW.writerow(testFields)

        self.trainWriter = tf.train.SummaryWriter(os.path.join(save, 'train'),
                                                  self.sess.graph)
        self.sess.run(tf.initialize_all_variables())

        nParams = np.sum(v.get_shape().num_elements() for v in tf.trainable_variables())

        meta = {'nTrain': nTrain, 'trainBatchSz': args.trainBatchSz,
                'nParams': nParams, 'nEpoch': args.nEpoch,
                'nIter': nIter}
        metaP = os.path.join(save, 'meta.json')
        with open(metaP, 'w') as f:
            json.dump(meta, f, indent=2)

        if not args.noncvx:
            self.sess.run(self.makeCvx)

        # Scale weights.
        # tflearn.is_training(True)
        # I = npr.randint(nTrain, size=10000)
        # xBatch = trainX[I, :]
        # yBatch = trainY[I, :]
        # W_z_zs = [x for x in self.theta_ if 'zu_proj/W' in x.name]
        # W_z_ys = [x for x in self.theta_ if '_yu/' in x.name]
        # W_z_us = [x for x in self.theta_ if re.match('z[0-9]*_u/W', x.name) is not None]
        # b_z = [x for x in self.theta_ if re.match('z[0-9]*_u/b', x.name) is not None]
        # z_zs, z_ys, z_us = self.sess.run([self.z_zs, self.z_ys, self.z_us],
        #                                  feed_dict={self.x_: xBatch, self.y_: yBatch})
        # print("=== Original ===")
        # new_bzs = []
        # scale = 0.1
        # for i in range(self.nLayers+1):
        #     sz = self.szs[i] if i < self.nLayers else 1
        #     b_zi_new = np.zeros(sz)

        #     z_ys_i_mean = np.mean(z_ys[i], axis=0)
        #     z_ys_i_std = np.std(z_ys[i], axis=0)
        #     print('y.std:', z_ys_i_std[:5])
        #     # self.sess.run(tf.assign(W_z_ys[i],
        #     #                         tf.matmul(W_z_ys[i], tf.diag(scale/z_ys_i_std))))
        #     # b_zi_new -= z_ys_i_mean/z_ys_i_std
        #     # b_zi_new -= z_ys_i_mean

        #     z_us_i_mean = np.mean(z_us[i], axis=0)
        #     z_us_i_std = np.std(z_us[i], axis=0)
        #     print('u.std:', z_us_i_std[:5])
        #     # self.sess.run(tf.assign(W_z_us[i],
        #     #                         tf.matmul(W_z_us[i], tf.diag(scale/z_us_i_std))))
        #     # b_zi_new -= z_us_i_mean/z_us_i_std
        #     # b_zi_new -= z_us_i_mean

        #     new_bzs.append(b_zi_new)
        #     if i == 0:
        #         self.sess.run(tf.assign(b_z[i], new_bzs[i]))

        # for i in range(self.nLayers):
        #     z_zs, = self.sess.run([self.z_zs],
        #                          feed_dict={self.x_: xBatch, self.y_: yBatch})
        #     z_zs_i_mean = np.mean(z_zs[i], axis=0)
        #     z_zs_i_std = np.std(z_zs[i], axis=0)
        #     print('z.std:', z_zs_i_std[:5])
        #     self.sess.run(tf.assign(W_z_zs[i],
        #                             tf.matmul(W_z_zs[i], tf.diag(scale/z_zs_i_std))))
        #     new_bzs[i+1] -= z_zs_i_mean/z_zs_i_std
        #     self.sess.run(tf.assign(b_z[i+1], new_bzs[i+1]))

        # print("=== Scaled ===")
        # z_zs, z_ys, z_us = self.sess.run([self.z_zs, self.z_ys, self.z_us],
        #                                  feed_dict={self.x_: xBatch, self.y_: yBatch})
        # for i in range(self.nLayers+1):
        #     if i > 0:
        #         z_zs_i_std = np.std(z_zs[i-1], axis=0)
        #         print('z.std:', z_zs_i_std[:5])

        #     z_ys_i_std = np.std(z_ys[i], axis=0)
        #     print('y.std:', z_ys_i_std[:5])

        #     z_us_i_std = np.std(z_us[i], axis=0)
        #     print('u.std:', z_us_i_std[:5])

        # import IPython; IPython.embed(); sys.exit(-1)
        # sys.exit(-1)

        bestTestF1 = 0.0
        nErrors = 0
        for i in range(nIter):
            tflearn.is_training(True)

            print("=== Iteration {} (Epoch {:.2f}) ===".format(
                i, i/np.ceil(nTrain/args.trainBatchSz)))
            start = time.time()
            I = npr.randint(nTrain, size=args.trainBatchSz)
            xBatch = trainX[I, :]
            yBatch = trainY[I, :]

            def fg(yhats):
                fd = {self.x_: xBatch, self.y_: yhats}
                e, ge = self.sess.run([self.E_, self.dE_dy_], feed_dict=fd)
                return e, ge

            y0 = np.full(yBatch.shape, 0.5)
            try:
                yN, G, h, lam, ys, nIters = bundle_entropy.solveBatch(fg, y0, nIter=10)
            except:
                print("Warning: Exception in bundle_entropy.solveBatch")
                nErrors += 1
                if nErrors > 10:
                    print("More than 10 errors raised, quitting")
                    sys.exit-(1-)
                continue

            nActive = [len(Gi) for Gi in G]
            l_yN = crossEntr(yBatch, yN)
            trainF1 = self.sess.run(self.F1_, feed_dict={self.y_: yN, self.trueY_: yBatch})

            fd = self.train_step_fd(args.trainBatchSz, xBatch, yBatch, G, yN, ys, lam)
            fd[self.l_yN_] = l_yN
            fd[self.nBundleIter_] = nIters
            fd[self.nActive_] = nActive
            summary, _ = self.sess.run([self.merged, self.train_step], feed_dict=fd)
            if not args.noncvx and len(self.proj) > 0:
                self.sess.run(self.proj)
            self.trainWriter.add_summary(summary, i)

            trainW.writerow((i, trainF1, l_yN))
            trainF.flush()

            print(" + trainF1: {:0.2f}".format(trainF1))
            print(" + loss: {:0.5e}".format(l_yN))
            print(" + time: {:0.2f} s".format(time.time()-start))

            # if i == 2000:
            #     tf.get_variable_scope().reuse_variables()
            #     W = tf.get_variable('W/yu')
            #     import matplotlib as mpl
            #     mpl.use('Agg')
            #     import matplotlib.pyplot as plt
            #     plt.style.use('bmh')
            #     import IPython; IPython.embed(); sys.exit(-1)

            if i % np.ceil(nTrain/args.trainBatchSz) == 0:
                print("=== Testing ===")
                tflearn.is_training(True)
                def fg(yhats):
                    fd = {self.x_: valX, self.y_: valY}
                    e, ge = self.sess.run([self.E_, self.dE_dy_], feed_dict=fd)
                    return e, ge

                y0 = np.full(valY.shape, 0.5)
                yN, G, h, lam, ys, _ = bundle_entropy.solveBatch(fg, y0, nIter=10)
                testF1 = self.sess.run(self.F1_,
                    feed_dict={self.y_: yN, self.trueY_: valY})
                l_yN = crossEntr(valY, yN)
                print(" + testF1: {:0.4f}".format(testF1))
                testW.writerow((i, testF1, l_yN))
                testF.flush()

                if testF1 > bestTestF1:
                    print('+ Saving best model.')
                    self.save(os.path.join(args.save, 'best.tf'))
                    bestTestF1 = testF1

                os.system('./icnn.plot.py ' + args.save)

        trainF.close()
        testF.close()

        meta['nErrors'] = nErrors
        with open(metaP, 'w') as f:
            json.dump(meta, f, indent=2)

        os.system('./icnn.plot.py ' + args.save)

    def save(self, path):
        self.saver.save(self.sess, path)

    def load(self, path):
        self.saver.restore(self.sess, path)

    def train_step_fd(self, trainBatchSz, xBatch, yBatch, G, yN, ys, lam):
        fd_xs, fd_ys, fd_vs, fd_cs = ([] for i in range(4))

        for j in range(trainBatchSz):
            Gj = np.array(G[j])
            cy, clam, ct = crossEntrGrad(yN[j], yBatch[j], Gj)
            for i in range(len(G[j])):
                fd_xs.append(xBatch[j])
                fd_ys.append(ys[j][i])
                v = lam[j][i] * cy + clam[i] * (yN[j] - ys[j][i])
                fd_vs.append(v)
                fd_cs.append(clam[i])

        fd_xs = np.array(fd_xs)
        fd_ys = np.array(fd_ys)
        fd_vs = np.array(fd_vs)
        fd_cs = np.array(fd_cs)
        fd = {self.x_: fd_xs, self.y_: fd_ys, self.v_: fd_vs, self.c_: fd_cs}
        return fd

    def f_shallow(self, x, y, szs, reuse=False):
        assert(len(szs) == 2)
        act = tf.nn.relu
        # act = tf.nn.softplus
        xy = tf.concat(1, (x, y))

        if reuse:
            tf.get_variable_scope().reuse_variables()

        # with tf.variable_scope('u0') as s:
        #     W_x = getW(self.nFeatures, szs[0], 'x')
        #     b = getB(szs[0])
        #     u0 = tf.identity(tf.matmul(x, W_x) + b, name='preact')
        #     tf.histogram_summary(u0.name, u0)
        #     u0 = act(u0, name='act')
        #     tf.histogram_summary(u0.name, u0)
        #     u0 = tflearn.layers.normalization.batch_normalization(u0, reuse=reuse,
        #                                                         scope=s, name='bn')
        #     tf.histogram_summary(u0.name, u0)

        # ui = u0
        # i,j = 0,1
        # with tf.variable_scope('u'+str(j)) as s:
        #     W_u = getW(szs[i], szs[j], 'u')
        #     b = getB(szs[j])
        #     uj = tf.identity(tf.matmul(ui, W_u) + b, name='preact')
        #     tf.histogram_summary(uj.name, uj)
        #     if len(szs) > i+2:
        #         uj = act(uj, name='act')
        #         tf.histogram_summary(uj.name, uj)
        #         uj = tflearn.layers.normalization.batch_normalization(uj, reuse=reuse,
        #                                                               scope=s, name='bn')
        #         tf.histogram_summary(uj.name, uj)

        # zSz = 10
        # with tf.variable_scope('z0'):
        #     z0 = act(tf.matmul(y, getW(self.nLabels, zSz, 'y')) + getB(zSz))
        # with tf.variable_scope('z1'):
        #     z1 = tf.reshape(tf.matmul(z0, getW(zSz, 1, 'z')) + getB(1), [-1])

        # zk = -tf.reduce_sum(tf.mul(x, y), 1)

        return tf.reshape(zk, [-1], name='energies')

    # def f(self, x, y, szs, reuse=False):
    #     assert(len(szs) == 2)
    #     act = tf.nn.relu
    #     # act = tf.nn.softplus
    #     xy = tf.concat(1, (x, y))

    #     if reuse:
    #         tf.get_variable_scope().reuse_variables()

    #     with tf.variable_scope('u0') as s:
    #         u0 = act(tf.matmul(x, getW(self.nFeatures, szs[0], 'x')))
    #         u0 = tflearn.layers.normalization.batch_normalization(u0, reuse=reuse,
    #                                                               scope=s, name='bn')
    #     with tf.variable_scope('u1') as s:
    #         u1 = tf.matmul(u0, getW(szs[0], szs[1], 'u')) + getB(szs[1])

    #     with tf.variable_scope('z0') as s:
    #         z0 = act(
    #             tf.matmul(y, getW(self.nLabels, self.nLabels, 'y')) + \
    #             getB(self.nLabels)
    #         )

    #     with tf.variable_scope('z1') as s:
    #         z1 = tf.add_n((
    #             tf.matmul(z0, getW_pos(self.nLabels, 1, 'z')),
    #             tf.matmul(y, getW(self.nLabels, 1, 'y')),
    #             tf.matmul(tf.mul(y, u1), getW(self.nLabels, 1, 'yu')),
    #             tf.matmul(
    #                 tf.mul(z0,
    #                        tf.matmul(act(u1),
    #                                  getW_pos(self.nLabels, self.nLabels, 'zu_inner'))),
    #                 getW_pos(self.nLabels, 1, 'zu')),
    #         )) + getB(1)

    #     # z1 = tf.matmul(tf.mul(y, x), getW(self.nLabels, 1, 'yu'))

    #     return tf.reshape(z1, [-1], name='energies')

    def f(self, x, y, szs, reuse=False):
        assert(len(szs) >= 1)
        fc = tflearn.fully_connected
        bn = tflearn.batch_normalization

        if szs[-1] != self.nLabels:
            print("Appending nLabels to layer sizes")
            szs.append(self.nLabels)

        self.szs = szs

        if reuse:
            tf.get_variable_scope().reuse_variables()

        self.nLayers = len(szs)
        self.us = []
        self.zs = []
        self.z_zs = []
        self.z_ys = []
        self.z_us = []

        reg = None #'L2'

        prevU = x
        for i in range(self.nLayers):
            with tf.variable_scope('u'+str(i)) as s:
                u = fc(prevU, szs[i], reuse=reuse, scope=s, regularizer=reg)
                if i < self.nLayers-1:
                    u = tf.nn.relu(u)
                    u = bn(u, reuse=reuse, scope=s, name='bn')
            self.us.append(u)
            prevU = u

        prevU, prevZ = x, y
        for i in range(self.nLayers+1):
            sz = szs[i] if i < self.nLayers else 1
            z_add = []
            if i > 0:
                with tf.variable_scope('z{}_zu_u'.format(i)) as s:
                    zu_u = fc(prevU, szs[i-1], reuse=reuse, scope=s,
                              activation='relu', bias=True, regularizer=reg)
                with tf.variable_scope('z{}_zu_proj'.format(i)) as s:
                    z_zu = fc(tf.mul(prevZ, zu_u), sz, reuse=reuse, scope=s,
                              bias=False, regularizer=reg)
                self.z_zs.append(z_zu)
                z_add.append(z_zu)

            with tf.variable_scope('z{}_yu_u'.format(i)) as s:
                yu_u = fc(prevU, self.nLabels, reuse=reuse, scope=s, bias=True,
                          regularizer=reg)
            with tf.variable_scope('z{}_yu'.format(i)) as s:
                z_yu = fc(tf.mul(y, yu_u), sz, reuse=reuse, scope=s, bias=False,
                          regularizer=reg)
                self.z_ys.append(z_yu)
            z_add.append(z_yu)

            with tf.variable_scope('z{}_u'.format(i)) as s:
                z_u = fc(prevU, sz, reuse=reuse, scope=s, bias=True, regularizer=reg)
            self.z_us.append(z_u)
            z_add.append(z_u)

            z = tf.add_n(z_add)
            variable_summaries(z, 'z{}_preact'.format(i))
            if i < self.nLayers:
                z = tf.nn.relu(z)
                variable_summaries(z, 'z{}_act'.format(i))

            self.zs.append(z)
            prevU = self.us[i] if i < self.nLayers else None
            prevZ = z

        z = tf.reshape(z, [-1], name='energies')
        return z

    def f_old(self, x, y, szs, reuse=False):
        assert(len(szs) >= 1)
        act = tf.nn.relu
        # act = tf.nn.softplus
        xy = tf.concat(1, (x, y))

        if reuse:
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope('z0') as s:
            W_xy = getW(self.nXy, szs[0], 'xy')
            b = getB(szs[0])
            z0 = tf.identity(tf.matmul(xy, W_xy) + b, name='preact')
            tf.histogram_summary(z0.name, z0)
            z0 = act(z0, name='act')
            tf.histogram_summary(z0.name, z0)

        with tf.variable_scope('u0') as s:
            W_x = getW(self.nFeatures, szs[0], 'x')
            b = getB(szs[0])
            u0 = tf.identity(tf.matmul(x, W_x) + b, name='preact')
            tf.histogram_summary(u0.name, u0)
            if len(szs) > 1:
                u0 = act(u0, name='act')
                tf.histogram_summary(u0.name, u0)
                u0 = tflearn.layers.normalization.batch_normalization(u0, reuse=reuse,
                                                                    scope=s, name='bn')
                tf.histogram_summary(u0.name, u0)

        zi, ui = z0, u0

        for i in range(len(szs)-1):
            j = i+1
            with tf.variable_scope('z'+str(j)) as s:
                zj_z = tf.matmul(zi, getW_pos(szs[i], szs[j], 'z'))
                zj_y = tf.matmul(y, getW(self.nLabels, szs[j], 'y'))
                zj_zu = tf.matmul(
                    tf.mul(zi, tf.matmul(act(ui), getW_pos(szs[i], szs[i], 'zu_inner'))),
                    getW_pos(szs[i], szs[j], 'zu'))
                zj_yu = tf.matmul(
                    tf.mul(y, tf.matmul(ui, getW(szs[i], self.nLabels, 'yu_inner'))),
                    getW(self.nLabels, szs[j], 'yu'))
                b = getB(szs[j])
                zj = tf.identity(zj_z + zj_y + zj_zu + zj_yu + b, name='preact')
                tf.histogram_summary(zj.name, zj)
                zj = act(zj, name='act')
                tf.histogram_summary(zj.name, zj)
                # z0 = slim.batch_norm(z0, is_training=True,
                                     # reuse=reuse, scope=s)

            with tf.variable_scope('u'+str(j)) as s:
                W_u = getW(szs[i], szs[j], 'u')
                b = getB(szs[j])
                uj = tf.identity(tf.matmul(ui, W_u) + b, name='preact')
                tf.histogram_summary(uj.name, uj)
                if len(szs) > i+2:
                    uj = act(uj, name='act')
                    tf.histogram_summary(uj.name, uj)
                    uj = tflearn.layers.normalization.batch_normalization(uj, reuse=reuse,
                                                                        scope=s, name='bn')
                    tf.histogram_summary(uj.name, uj)

            zi, ui = zj, uj

        i = len(szs)-1
        j = i+1
        with tf.variable_scope('z'+str(j)):
            zj_z = tf.matmul(zi, getW_pos(szs[i], 1, 'z'))
            zj_u = tf.matmul(ui, getW(szs[i], 1, 'u'))
            zj_y = tf.matmul(y, getW(self.nLabels, 1, 'y'))
            zj_zu = tf.matmul(
                tf.mul(zi, tf.matmul(act(ui), getW_pos(szs[i], szs[i], 'zu_inner'))),
                getW_pos(szs[i], 1, 'zu'))
            zj_yu = tf.matmul(
                tf.mul(y, tf.matmul(ui, getW(szs[i], self.nLabels, 'yu_inner'))),
                getW(self.nLabels, 1, 'yu'))
            b = getB(1)
            zj = zj_z + zj_u + zj_y + zj_zu + zj_yu + b
            tf.histogram_summary(zj.name, zj)

        return tf.reshape(zj, [-1], name='energies')


def tf_nOnes(b):
    # Must be binary.
    return tf.reduce_sum(tf.cast(b, tf.int32))

def crossEntrGrad(y, trueY, G):
    k,n = G.shape
    assert(len(y) == n)
    y_ = np.copy(y)
    eps = 1e-8
    y_ = np.clip(y_, eps, 1.-eps)

    # H = np.bmat([[np.diag(1./y_ + 1./(1.-y_)), G.T, np.zeros((n,1))],
    #              [G, np.zeros((k,k)), -np.ones((k,1))],
    #              [np.zeros((1,n)), -np.ones((1,k)), np.zeros((1,1))]])

    # c = -np.linalg.solve(H, np.concatenate([trueY/y_-(1-trueY)/(1-y_), np.zeros(k+1)]))
    # b = np.concatenate([trueY/y_-(1-trueY)/(1-y_), np.zeros(k+1)])
    # cy, clam, ct = np.split(c, [n, n+k])
    # cy[(y == 0) | (y == 1)] = 0

    z = 1./y_ + 1./(1.-y_)
    zinv = 1./z
    G_zinv = G*zinv
    G_zinv_GT = np.dot(G_zinv, G.T)
    H = np.bmat([[G_zinv_GT, np.ones((k,1))], [np.ones((1,k)), np.zeros((1,1))]])
    dl = trueY/y_-(1-trueY)/(1-y_)
    b = np.concatenate([np.dot(G_zinv, dl), np.zeros(1)])
    clamt = np.linalg.solve(H, b)
    clam, ct = np.split(clamt, [k])
    cy = zinv*dl - np.dot((G*zinv).T, clam)
    cy[(y == 0) | (y == 1)] = 0
    return cy, clam, ct

def crossEntr(trueY, y):
    l = -np.sum((trueY * np.log(y))[y>0]) -np.sum(((1.-trueY)*np.log(1.-y))[y<1])
    return l

if __name__=='__main__':
    main()
