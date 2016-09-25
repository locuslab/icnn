#!/usr/bin/env python3

import tensorflow as tf
import tflearn
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import array_ops

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

import setproctitle

from datetime import datetime

sys.path.append('../lib')

import olivetti
import bundle_entropy

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='work/mse.ebundle')
    parser.add_argument('--nEpoch', type=float, default=50)
    parser.add_argument('--nBundleIter', type=int, default=30)
    # parser.add_argument('--trainBatchSz', type=int, default=25)
    parser.add_argument('--trainBatchSz', type=int, default=70)
    # parser.add_argument('--testBatchSz', type=int, default=2048)
    parser.add_argument('--noncvx', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--valSplit', type=float, default=0)

    args = parser.parse_args()

    assert(not args.noncvx)

    setproctitle.setproctitle('bamos.icnn.comp.mse.ebundle')

    npr.seed(args.seed)
    tf.set_random_seed(args.seed)

    save = os.path.expanduser(args.save)
    if os.path.isdir(save):
        shutil.rmtree(save)
    os.makedirs(save)
    ckptDir = os.path.join(save, 'ckpt')
    args.ckptDir = ckptDir
    if not os.path.exists(ckptDir):
        os.makedirs(ckptDir)

    data = olivetti.load("data/olivetti")
    # eps = 1e-8
    # data['trainX'] = data['trainX'].clip(eps, 1.-eps)
    # data['trainY'] = data['trainY'].clip(eps, 1.-eps)
    # data['testX'] = data['testX'].clip(eps, 1.-eps)
    # data['testY'] = data['testY'].clip(eps, 1.-eps)

    nTrain = data['trainX'].shape[0]
    nTest = data['testX'].shape[0]

    inputSz = list(data['trainX'][0].shape)
    outputSz = list(data['trainY'][1].shape)

    print("\n\n" + "="*40)
    print("+ nTrain: {}, nTest: {}".format(nTrain, nTest))
    print("+ inputSz: {}, outputSz: {}".format(inputSz, outputSz))
    print("="*40 + "\n\n")

    config = tf.ConfigProto() #log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = Model(inputSz, outputSz, sess)
        model.train(args, data['trainX'], data['trainY'], data['testX'], data['testY'])

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

class Model:
    def __init__(self, inputSz, outputSz, sess):
        self.inputSz = inputSz
        self.outputSz = outputSz
        self.nOutput = np.prod(outputSz)
        self.sess = sess

        self.trueY_ = tf.placeholder(tf.float32, shape=[None] + outputSz, name='trueY')

        self.x_ = tf.placeholder(tf.float32, shape=[None] + inputSz, name='x')
        self.y_ = tf.placeholder(tf.float32, shape=[None] + outputSz, name='y')
        self.v_ = tf.placeholder(tf.float32, shape=[None, self.nOutput], name='v')
        self.c_ = tf.placeholder(tf.float32, shape=[None], name='c')

        self.E_ = self.f(self.x_, self.y_)
        variable_summaries(self.E_)
        self.dE_dy_ = tf.gradients(self.E_, self.y_)[0]
        self.dE_dyFlat_ = tf.contrib.layers.flatten(self.dE_dy_)

        self.yFlat_ = tf.contrib.layers.flatten(self.y_)
        self.E_entr_ = self.E_ + tf.reduce_sum(self.yFlat_*tf.log(self.yFlat_), 1) + \
                       tf.reduce_sum((1.-self.yFlat_)*tf.log(1.-self.yFlat_), 1)
        self.dE_entr_dy_ = tf.gradients(self.E_entr_, self.y_)[0]
        self.dE_entr_dyFlat_ = tf.contrib.layers.flatten(self.dE_entr_dy_)

        self.F_ = tf.mul(self.c_, self.E_) + \
                  tf.reduce_sum(tf.mul(self.dE_dyFlat_, self.v_), 1)

        # regLosses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # self.F_reg_ = self.F_ + 0.1*regLosses
        # self.F_reg_ = self.F_ + 1e-5*tf.square(self.E_)

        self.opt = tf.train.AdamOptimizer(0.001)
        self.theta_ = tf.trainable_variables()
        self.gv_ = [(g,v) for g,v in self.opt.compute_gradients(self.F_, self.theta_)
                    if g is not None]
        self.train_step = self.opt.apply_gradients(self.gv_)

        self.theta_cvx_ = [v for v in self.theta_
                           if 'proj' in v.name and 'W:' in v.name]

        self.makeCvx = [v.assign(tf.abs(v)/2.0) for v in self.theta_cvx_]
        self.proj = [v.assign(tf.maximum(v, 0)) for v in self.theta_cvx_]

        for g,v in self.gv_:
            variable_summaries(g, 'gradients/'+v.name)

        self.l_yN_ = tf.placeholder(tf.float32, name='l_yN')
        tf.scalar_summary('mse', self.l_yN_)

        self.nBundleIter_ = tf.placeholder(tf.float32, [None], name='nBundleIter')
        variable_summaries(self.nBundleIter_)

        self.nActive_ = tf.placeholder(tf.float32, [None], name='nActive')
        variable_summaries(self.nActive_)

        self.merged = tf.merge_all_summaries()
        self.saver = tf.train.Saver(max_to_keep=0)


    def train(self, args, trainX, trainY, valX, valY):
        save = args.save

        self.meanY = np.mean(trainY, axis=0)

        nTrain = trainX.shape[0]
        nTest = valX.shape[0]

        nIter = int(np.ceil(args.nEpoch*nTrain/args.trainBatchSz))

        trainFields = ['iter', 'loss']
        trainF = open(os.path.join(save, 'train.csv'), 'w')
        trainW = csv.writer(trainF)
        trainW.writerow(trainFields)
        trainF.flush()

        testFields = ['iter', 'loss']
        testF = open(os.path.join(save, 'test.csv'), 'w')
        testW = csv.writer(testF)
        testW.writerow(testFields)
        testF.flush()

        self.trainWriter = tf.train.SummaryWriter(os.path.join(save, 'train'),
                                                  self.sess.graph)
        self.sess.run(tf.initialize_all_variables())
        if not args.noncvx:
            self.sess.run(self.makeCvx)

        nParams = np.sum(v.get_shape().num_elements() for v in tf.trainable_variables())

        self.nBundleIter = args.nBundleIter
        meta = {'nTrain': nTrain, 'trainBatchSz': args.trainBatchSz,
                'nParams': nParams, 'nEpoch': args.nEpoch,
                'nIter': nIter, 'nBundleIter': self.nBundleIter}
        metaP = os.path.join(save, 'meta.json')
        with open(metaP, 'w') as f:
            json.dump(meta, f, indent=2)

        nErrors = 0
        maxErrors = 20
        for i in range(nIter):
            tflearn.is_training(True)

            print("=== Iteration {} (Epoch {:.2f}) ===".format(
                i, i/np.ceil(nTrain/args.trainBatchSz)))
            start = time.time()
            I = npr.randint(nTrain, size=args.trainBatchSz)
            xBatch = trainX[I, :]
            yBatch = trainY[I, :]
            yBatch_flat = yBatch.reshape((args.trainBatchSz, -1))

            xBatch_flipped = xBatch[:,:,::-1,:]

            def fg(yhats):
                yhats_shaped = yhats.reshape([args.trainBatchSz]+self.outputSz)
                fd = {self.x_: xBatch_flipped, self.y_: yhats_shaped}
                e, ge = self.sess.run([self.E_, self.dE_dyFlat_], feed_dict=fd)
                return e, ge

            y0 = np.expand_dims(self.meanY, axis=0).repeat(args.trainBatchSz, axis=0)
            y0 = y0.reshape((args.trainBatchSz, -1))
            try:
                yN, G, h, lam, ys, nIters = bundle_entropy.solveBatch(
                    fg, y0, nIter=self.nBundleIter)
                yN_shaped = yN.reshape([args.trainBatchSz]+self.outputSz)
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                print("Warning: Exception in bundle_entropy.solveBatch")
                nErrors += 1
                if nErrors > maxErrors:
                    print("More than {} errors raised, quitting".format(maxErrors))
                    sys.exit(-1)
                continue

            nActive = [len(Gi) for Gi in G]
            l_yN = mse(yBatch_flat, yN)

            fd = self.train_step_fd(args.trainBatchSz, xBatch_flipped, yBatch_flat,
                                    G, yN, ys, lam)
            fd[self.l_yN_] = l_yN
            fd[self.nBundleIter_] = nIters
            fd[self.nActive_] = nActive
            summary, _ = self.sess.run([self.merged, self.train_step], feed_dict=fd)
            if not args.noncvx and len(self.proj) > 0:
                self.sess.run(self.proj)

            saveImgs(xBatch, yN_shaped, "{}/trainImgs/{:05d}".format(args.save, i))

            self.trainWriter.add_summary(summary, i)

            trainW.writerow((i, l_yN))
            trainF.flush()

            print(" + loss: {:0.5e}".format(l_yN))
            print(" + time: {:0.2f} s".format(time.time()-start))

            if i % np.ceil(nTrain/(4.0*args.trainBatchSz)) == 0:
                os.system('./icnn.plot.py ' + args.save)

            if i % np.ceil(nTrain/args.trainBatchSz) == 0:
                print("=== Testing ===")
                tflearn.is_training(False)

                y0 = np.expand_dims(self.meanY, axis=0).repeat(nTest, axis=0)
                y0 = y0.reshape((nTest, -1))
                valX_flipped = valX[:,:,::-1,:]

                def fg(yhats):
                    yhats_shaped = yhats.reshape([nTest]+self.outputSz)
                    fd = {self.x_: valX_flipped, self.y_: yhats_shaped}
                    e, ge = self.sess.run([self.E_, self.dE_dyFlat_], feed_dict=fd)
                    return e, ge

                try:
                    yN, G, h, lam, ys, nIters = bundle_entropy.solveBatch(
                        fg, y0, nIter=self.nBundleIter)
                    yN_shaped = yN.reshape([nTest]+self.outputSz)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except:
                    print("Warning: Exception in bundle_entropy.solveBatch")
                    nErrors += 1
                    if nErrors > maxErrors:
                        print("More than {} errors raised, quitting".format(maxErrors))
                        sys.exit(-1)
                    continue

                testMSE = mse(valY, yN_shaped)

                saveImgs(valX, yN_shaped, "{}/testImgs/{:05d}".format(args.save, i))

                print(" + test loss: {:0.5e}".format(testMSE))
                testW.writerow((i, testMSE))
                testF.flush()

                self.save(os.path.join(args.ckptDir, '{:05d}.tf'.format(i)))

                os.system('./icnn.plot.py ' + args.save)

        trainF.close()
        testF.close()

        os.system('./icnn.plot.py ' + args.save)

    def save(self, path):
        self.saver.save(self.sess, path)

    def load(self, path):
        self.saver.restore(self.sess, path)

    def train_step_fd(self, trainBatchSz, xBatch, yBatch, G, yN, ys, lam):
        fd_xs, fd_ys, fd_vs, fd_cs = ([] for i in range(4))

        for j in range(trainBatchSz):
            if len(G[j]) == 0:
                continue
            Gj = np.array(G[j])
            cy, clam, ct = mseGrad(yN[j], yBatch[j], Gj)
            for i in range(len(G[j])):
                fd_xs.append(xBatch[j])
                fd_ys.append(ys[j][i].reshape(self.outputSz))
                v = lam[j][i] * cy + clam[i] * (yN[j] - ys[j][i])
                fd_vs.append(v)
                fd_cs.append(clam[i])

        fd_xs = np.array(fd_xs)
        fd_ys = np.array(fd_ys)
        fd_vs = np.array(fd_vs)
        fd_cs = np.array(fd_cs)
        fd = {self.x_: fd_xs, self.y_: fd_ys, self.v_: fd_vs, self.c_: fd_cs}
        return fd

    def f(self, x, y, reuse=False):
        conv = tflearn.conv_2d
        bn = tflearn.batch_normalization
        fc = tflearn.fully_connected

        # Architecture from 'Human-level control through deep reinforcement learning'
        # http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html
        convs = [(32, 8, [1,4,4,1]), (64, 4, [1,2,2,1]), (64, 3, [1,1,1,1])]
        fcs = [512, 1]

        reg = None #'L2'

        us = []
        zs = []

        layerI = 0
        prevU = x
        for nFilter, kSz, strides in convs:
            with tf.variable_scope('u'+str(layerI)) as s:
                u = bn(conv(prevU, nFilter, kSz, strides=strides, activation='relu',
                            scope=s, reuse=reuse, regularizer=reg),
                       scope=s, reuse=reuse)
            us.append(u)
            prevU = u
            layerI += 1

        for sz in fcs:
            with tf.variable_scope('u'+str(layerI)) as s:
                u = fc(prevU, sz, scope=s, reuse=reuse, regularizer=reg)
                if sz == 1:
                    u = tf.reshape(u, [-1])
                else:
                    u = bn(tf.nn.relu(u), scope=s, reuse=reuse)
            us.append(u)
            prevU = u
            layerI += 1

        layerI = 0
        prevU, prevZ, y_red = x, None, y
        for nFilter, kSz, strides in convs:
            z_add = []
            if layerI > 0:
                with tf.variable_scope('z{}_zu_u'.format(layerI)) as s:
                    prev_nFilter = convs[layerI-1][0]
                    zu_u = conv(prevU, prev_nFilter, 3, reuse=reuse,
                                scope=s, activation='relu', bias=True, regularizer=reg)
                with tf.variable_scope('z{}_zu_proj'.format(layerI)) as s:
                    z_zu = conv(tf.mul(prevZ, zu_u), nFilter, kSz, strides=strides,
                                reuse=reuse, scope=s, bias=False, regularizer=reg)
                z_add.append(z_zu)

            with tf.variable_scope('z{}_yu_u'.format(layerI)) as s:
                yu_u = conv(prevU, 1, 3, reuse=reuse, scope=s,
                            bias=True, regularizer=reg)
            with tf.variable_scope('z{}_yu'.format(layerI)) as s:
                z_yu = conv(tf.mul(y_red, yu_u), nFilter, kSz, strides=strides,
                            reuse=reuse, scope=s, bias=False, regularizer=reg)
            with tf.variable_scope('z{}_y_red'.format(layerI)) as s:
                y_red = conv(y_red, 1, kSz, strides=strides, reuse=reuse,
                             scope=s, bias=True, regularizer=reg)
            z_add.append(z_yu)

            with tf.variable_scope('z{}_u'.format(layerI)) as s:
                z_u = conv(prevU, nFilter, kSz, strides=strides, reuse=reuse,
                           scope=s, bias=True, regularizer=reg)
            z_add.append(z_u)

            z = tf.nn.relu(tf.add_n(z_add))

            zs.append(z)
            prevU = us[layerI] if layerI < len(us) else None
            prevZ = z
            layerI += 1

        prevZ = tf.contrib.layers.flatten(prevZ)
        prevU = tf.contrib.layers.flatten(prevU)
        y_red_flat = tf.contrib.layers.flatten(y_red)
        for sz in fcs:
            z_add = []
            with tf.variable_scope('z{}_zu_u'.format(layerI)) as s:
                prevU_sz = prevU.get_shape()[1].value
                zu_u = fc(prevU, prevU_sz, reuse=reuse, scope=s,
                            activation='relu', bias=True, regularizer=reg)
            with tf.variable_scope('z{}_zu_proj'.format(layerI)) as s:
                z_zu = fc(tf.mul(prevZ, zu_u), sz, reuse=reuse, scope=s,
                            bias=False, regularizer=reg)
            z_add.append(z_zu)

            # y passthrough in the FC layers:
            #
            # with tf.variable_scope('z{}_yu_u'.format(layerI)) as s:
            #     ycf_sz = y_red_flat.get_shape()[1].value
            #     yu_u = fc(prevU, ycf_sz, reuse=reuse, scope=s, bias=True,
            #               regularizer=reg)
            # with tf.variable_scope('z{}_yu'.format(layerI)) as s:
            #     z_yu = fc(tf.mul(y_red_flat, yu_u), sz, reuse=reuse, scope=s,
            #               bias=False, regularizer=reg)
            # z_add.append(z_yu)

            with tf.variable_scope('z{}_u'.format(layerI)) as s:
                z_u = fc(prevU, sz, reuse=reuse, scope=s, bias=True, regularizer=reg)
            z_add.append(z_u)

            z = tf.add_n(z_add)
            variable_summaries(z, 'z{}_preact'.format(layerI))
            if sz != 1:
                z = tf.nn.relu(z)
                variable_summaries(z, 'z{}_act'.format(layerI))

            prevU = us[layerI] if layerI < len(us) else None
            prevZ = z
            zs.append(z)
            layerI += 1

        z = tf.reshape(z, [-1], name='energies')
        return z

def saveImgs(xs, ys, save, colWidth=10):
    nImgs = xs.shape[0]
    assert(nImgs == ys.shape[0])

    if not os.path.exists(save):
        os.makedirs(save)

    fnames = []
    for i in range(nImgs):
        xy = np.clip(np.squeeze(np.concatenate([ys[i], xs[i]], axis=1)), 0., 1.)
        # Imagemagick montage has intensity scaling issues with png output files here.
        fname = "{}/{:04d}.jpg".format(save, i)
        plt.imsave(fname, xy, cmap=mpl.cm.gray)
        fnames.append(fname)

    os.system('montage -geometry +0+0 -tile {}x {} {}.png'.format(
        colWidth, ' '.join(fnames), save))

def tf_nOnes(b):
    # Must be binary.
    return tf.reduce_sum(tf.cast(b, tf.int32))

def mse(y, trueY):
    return np.mean(np.square(255.*(y-trueY)))
    # return 0.5*np.sum(np.square((y-trueY)))

def mseGrad_full(y, trueY, G):
    k,n = G.shape
    assert(len(y) == n)
    I = np.where((y > 1e-8) & (1.-y > 1e-8))
    z = np.ones_like(y)
    z[I] = (1./y[I] + 1./(1.-y[I]))
    H = np.bmat([[np.diag(z), G.T, np.zeros((n,1))],
                 [G, np.zeros((k,k)), -np.ones((k,1))],
                 [np.zeros((1,n)), -np.ones((1,k)), np.zeros((1,1))]])

    c = -np.linalg.solve(H, np.concatenate([(y - trueY), np.zeros(k+1)]))
    return np.split(c, [n, n+k])

def mseGrad(y, trueY, G):
    try:
        k,n = G.shape
    except:
        import IPython; IPython.embed(); sys.exit(-1)
    assert(len(y) == n)

    # y_ = np.copy(y)
    # eps = 1e-8
    # y_ = np.clip(y_, eps, 1.-eps)

    I = np.where((y > 1e-8) & (1.-y > 1e-8))
    # print(len(I[0]))
    z = np.ones_like(y)
    z[I] = (1./y[I] + 1./(1.-y[I]))
    z = 1./y + 1./(1.-y)
    zinv = 1./z
    G_zinv = G*zinv
    G_zinv_GT = np.dot(G_zinv, G.T)
    H = np.bmat([[G_zinv_GT, np.ones((k,1))], [np.ones((1,k)), np.zeros((1,1))]])

    # Different scaling than the MSE plots.
    dl = -(y-trueY)

    b = np.concatenate([np.dot(G_zinv, dl), np.zeros(1)])
    clamt = np.linalg.solve(H, b)
    clam, ct = np.split(clamt, [k])
    cy = zinv*dl - np.dot((G*zinv).T, clam)
    cy[(y == 0) | (y == 1)] = 0
    return cy, clam, ct

if __name__=='__main__':
    main()
