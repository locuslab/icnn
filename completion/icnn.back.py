#!/usr/bin/env python3

import setGPU

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

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
# import skimage.io
# import scipy
# from scipy.misc import imsave
# import scipy

# @tf.python.ops.RegisterGradient("MaxPoolGrad")
# def _MaxPoolGradGrad(op, grad):
#     x = op.inputs[1]
#     return (gen_nn_ops._max_pool_grad(grad, x),
#             array_ops.zeros(shape=array_ops.shape(x), dtype=x.dtype))

# @tf.python.ops.RegisterGradient("AvgPoolGrad")
# def _AvgPoolGradGrad(op, grad):
#     x = op.inputs[1]
#     inShape = array_ops.shape(x)
#     ksize = op.get_attr('ksize')
#     strides = op.get_attr('strides')
#     padding = op.get_attr('padding')
#     # print(op)
#     import IPython; IPython.embed(); sys.exit(-1)
#     return (gen_nn_ops._avg_pool_grad(inShape, grad, ksize, strides, padding),
#             array_ops.zeros(shape=inShape, dtype=x.dtype))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='work/mse')
    parser.add_argument('--nEpoch', type=float, default=50)
    # parser.add_argument('--trainBatchSz', type=int, default=25)
    parser.add_argument('--trainBatchSz', type=int, default=70)
    # parser.add_argument('--testBatchSz', type=int, default=2048)
    parser.add_argument('--nGdIter', type=int, default=30)
    parser.add_argument('--noncvx', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--valSplit', type=float, default=0)

    args = parser.parse_args()

    setproctitle.setproctitle('bamos.icnn.comp.mse')

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
        model = Model(inputSz, outputSz, sess, args.nGdIter)
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
    def __init__(self, inputSz, outputSz, sess, nGdIter):
        self.inputSz = inputSz
        self.outputSz = outputSz
        self.sess = sess

        self.trueY_ = tf.placeholder(tf.float32, shape=[None] + outputSz, name='trueY')

        self.x_ = tf.placeholder(tf.float32, shape=[None] + inputSz, name='x')
        self.y0_ = tf.placeholder(tf.float32, shape=[None] + outputSz, name='y')

        E0_ = self.f(self.x_, self.y0_)

        lr = 0.01
        momentum = 0.9

        yi_ = self.y0_
        Ei_ = E0_
        vi_ = 0

        for i in range(nGdIter):
            prev_vi_ = vi_
            vi_ = momentum*prev_vi_ - lr*tf.gradients(Ei_, yi_)[0]
            yi_ = yi_ - momentum*prev_vi_ + (1.+momentum)*vi_
            Ei_ = self.f(self.x_, yi_, True)

        self.yn_ = yi_
        self.energies_ = Ei_

        self.mse_ = tf.reduce_mean(tf.square(255.*(self.yn_ - self.trueY_)))

        self.opt = tf.train.AdamOptimizer(0.001)
        self.theta_ = tf.trainable_variables()
        self.gv_ = [(g,v) for g,v in
                    self.opt.compute_gradients(self.mse_, self.theta_)
                    if g is not None]
        self.train_step = self.opt.apply_gradients(self.gv_)

        # print([x.name for x in self.theta_])
        # import IPython; IPython.embed(); sys.exit(-1)

        self.theta_cvx_ = [v for v in self.theta_
                           if 'proj' in v.name and 'W:' in v.name]

        self.makeCvx = [v.assign(tf.abs(v)/2.) for v in self.theta_cvx_]
        self.proj = [v.assign(tf.maximum(v, 0)) for v in self.theta_cvx_]

        # for g,v in self.gv_:
        #     variable_summaries(g, 'gradients/'+v.name)

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

        testFields = ['iter', 'loss']
        testF = open(os.path.join(save, 'test.csv'), 'w')
        testW = csv.writer(testF)
        testW.writerow(testFields)

        self.trainWriter = tf.train.SummaryWriter(os.path.join(save, 'train'),
                                                  self.sess.graph)
        self.sess.run(tf.initialize_all_variables())
        if not args.noncvx:
            self.sess.run(self.makeCvx)

        nParams = np.sum(v.get_shape().num_elements() for v in tf.trainable_variables())

        meta = {'nTrain': nTrain, 'trainBatchSz': args.trainBatchSz,
                'nParams': nParams, 'nEpoch': args.nEpoch,
                'nIter': nIter}
        metaP = os.path.join(save, 'meta.json')
        with open(metaP, 'w') as f:
            json.dump(meta, f, indent=2)

        # bestTestF1 = 0.0
        # nErrors = 0
        for i in range(nIter):
            tflearn.is_training(True)

            print("=== Iteration {} (Epoch {:.2f}) ===".format(
                i, i/np.ceil(nTrain/args.trainBatchSz)))
            start = time.time()
            I = npr.randint(nTrain, size=args.trainBatchSz)
            xBatch = trainX[I, :]
            yBatch = trainY[I, :]

            xBatch_flipped = xBatch[:,:,::-1,:]

            y0 = np.expand_dims(self.meanY, axis=0).repeat(args.trainBatchSz, axis=0)
            assert(y0.shape == yBatch.shape)
            _, trainMSE, yn = self.sess.run(
                [self.train_step, self.mse_, self.yn_],
                feed_dict={self.x_: xBatch_flipped, self.y0_: y0, self.trueY_: yBatch})
            if not args.noncvx and len(self.proj) > 0:
                self.sess.run(self.proj)

            saveImgs(xBatch, yn, "{}/trainImgs/{:05d}".format(args.save, i))

            # self.trainWriter.add_summary(summary, i)

            trainW.writerow((i, trainMSE))
            trainF.flush()

            print(" + loss: {:0.5e}".format(trainMSE))
            print(" + time: {:0.2f} s".format(time.time()-start))

            if i % np.ceil(nTrain/args.trainBatchSz) == 0:
                print("=== Testing ===")
                tflearn.is_training(False)
                # y0 = np.full(valY.shape, 0.5)
                y0 = np.expand_dims(self.meanY, axis=0).repeat(nTest, axis=0)
                assert(y0.shape == valY.shape)
                valX_flipped = valX[:,:,::-1,:]
                testMSE, yn = self.sess.run([self.mse_, self.yn_],
                    feed_dict={self.x_: valX_flipped, self.y0_: y0, self.trueY_: valY})

                saveImgs(valX, yn, "{}/testImgs/{:05d}".format(args.save, i))

                print(" + test loss: {:0.5e}".format(testMSE))
                testW.writerow((i, testMSE))
                testF.flush()

                # if testF1 > bestTestF1:
                #     print('+ Saving best model.')
                #     self.save(os.path.join(args.save, 'best.tf'))
                #     bestTestF1 = testF1

                os.system('./icnn.plot.py ' + args.save)

                self.save(os.path.join(args.ckptDir, '{:05d}.tf'.format(i)))

        trainF.close()
        testF.close()

        os.system('./icnn.plot.py ' + args.save)

    def save(self, path):
        self.saver.save(self.sess, path)

    def load(self, path):
        self.saver.restore(self.sess, path)

    def f(self, x, y, reuse=False):
        conv = tflearn.conv_2d
        bn = tflearn.batch_normalization
        fc = tflearn.fully_connected

        # Original DQN:
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
            # dsW_ = tf.constant(np.ones((strides[1], strides[2], 1, 1))/ \
                               # (strides[1]*strides[2]), dtype=tf.float32)
            # y_red = tf.nn.conv2d(y_red, dsW_, strides=strides, padding='SAME')
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

        # for i in range(len(us)):
        #     print(us[i].get_shape())
        # for i in range(len(zs)):
        #     print(zs[i].get_shape())

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

if __name__=='__main__':
    main()
