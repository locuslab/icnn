#!/usr/bin/env python3.4

import tensorflow as tf
import tflearn

import numpy as np
import numpy.random as npr

np.set_printoptions(precision=2)
np.seterr(all='raise')

import argparse
import csv
import os
import sys
import time
import pickle
import json

from datetime import datetime

import bibsonomy
import util

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='work/icnn')
    parser.add_argument('--nEpoch', type=int, default=100)
    parser.add_argument('--trainBatchSz', type=int, default=128)
    parser.add_argument('--layerSizes', type=int, nargs='+', default=[600, 600])
    # parser.add_argument('--testBatchSz', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data', type=str)
    parser.add_argument('--valSplit', type=float, default=0)
    parser.add_argument('--noncvx', action='store_true')
    parser.add_argument('--inference_lr', type=float, default=0.01)
    parser.add_argument('--inference_momentum', type=float, default=0.5)
    parser.add_argument('--inference_nIter', type=int, default=10)

    args = parser.parse_args()

    npr.seed(args.seed)
    tf.set_random_seed(args.seed)

    save = os.path.expanduser(args.save)
    if not os.path.isdir(save):
        os.makedirs(save, exist_ok=True)

    if args.data:
        print("Loading data from: ", args.data)
        with open(args.data, 'rb') as f:
            data = pickle.load(f)
    else:
        data = bibsonomy.loadBibtex("data/bibtex")

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
        model = Model(nFeatures, nLabels, args, sess)
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

class Model:
    def __init__(self, nFeatures, nLabels, args, sess):
        self.nFeatures = nFeatures
        self.nLabels = nLabels
        self.nXy = nFeatures + nLabels
        self.sess = sess

        self.trueY_ = tf.placeholder(tf.float32, shape=(None, nLabels), name='trueY')

        self.x_ = tf.placeholder(tf.float32, shape=(None, nFeatures), name='x')
        self.y0_ = tf.placeholder(tf.float32, shape=(None, nLabels), name='y')
        self.E0_ = self.f(self.x_, self.y0_, args.layerSizes)

        lr = args.inference_lr
        momentum = args.inference_momentum
        nIter = args.inference_nIter

        yi_ = self.y0_
        Ei_ = self.E0_
        vi_ = 0

        for i in range(nIter):
            prev_vi_ = vi_
            vi_ = momentum*prev_vi_ - lr*tf.gradients(Ei_, yi_)[0]
            yi_ = yi_ - momentum*prev_vi_ + (1.+momentum)*vi_
            Ei_ = self.f(self.x_, yi_, args.layerSizes, True)

        self.yn_ = yi_
        self.energies_ = Ei_

        self.mse_ = tf.reduce_mean(tf.square(self.yn_ - self.trueY_))

        self.opt = tf.train.AdamOptimizer(0.001)
        self.theta_ = tf.trainable_variables()
        gv_ = [(g,v) for g,v in self.opt.compute_gradients(self.mse_, self.theta_)
               if g is not None]
        self.train_step = self.opt.apply_gradients(gv_)

        self.theta_cvx_ = [v for v in self.theta_
                           if 'proj' in v.name and 'W:' in v.name]
        self.makeCvx = [v.assign(tf.abs(v)) for v in self.theta_cvx_]
        self.proj = [v.assign(tf.maximum(v, 0)) for v in self.theta_cvx_]

        self.merged = tf.merge_all_summaries()
        self.saver = tf.train.Saver(max_to_keep=1)

    def train(self, args, trainX, trainY, valX, valY):
        save = args.save

        nTrain = trainX.shape[0]
        nTest = valX.shape[0]

        nIter = int(args.nEpoch*np.ceil(nTrain/args.trainBatchSz))

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

        bestTestF1 = 0.0
        nErrors = 0
        for i in range(nIter):
            tflearn.is_training(True)

            print("=== Iteration {} (Epoch {:.2f}) ===".format(
                i, i/np.ceil(nTrain/args.trainBatchSz)))
            I = npr.randint(nTrain, size=args.trainBatchSz)
            xBatch = trainX[I, :]
            yBatch = trainY[I, :]


            y0 = np.full(yBatch.shape, 0.5)
            _, trainMSE, yN = self.sess.run(
                [self.train_step, self.mse_, self.yn_],
                feed_dict={self.x_: xBatch, self.y0_: y0, self.trueY_: yBatch})
            trainF1 = util.macroF1(yBatch, yN)
            self.sess.run(self.proj)

            trainW.writerow((i, trainF1, trainMSE))
            trainF.flush()

            print(" + trainF1: {:0.2f}".format(trainF1))
            print(" + MSE: {:0.2e}".format(trainMSE))

            if (i+1) % np.ceil(nTrain/args.trainBatchSz) == 0:
                print("=== Testing ===")
                tflearn.is_training(False)
                y0 = np.full(valY.shape, 0.5)
                yN = self.sess.run(self.yn_, feed_dict={self.x_: valX, self.y0_: y0})
                testF1 = util.macroF1(valY, yN)
                print(" + testF1: {:0.4f}".format(testF1))
                testW.writerow((i, testF1))
                testF.flush()

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

if __name__=='__main__':
    main()
