#!/usr/bin/env python3

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
import shutil

from datetime import datetime

import matplotlib as mpl
from matplotlib import cm
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

from sklearn.utils import shuffle
from sklearn.datasets import make_moons, make_circles, make_classification

import setproctitle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='work')
    parser.add_argument('--nEpoch', type=int, default=100)
    # parser.add_argument('--testBatchSz', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default="picnn",
                        choices=['picnn', 'ficnn'])
    parser.add_argument('--dataset', type=str, default="moons",
                        choices=['moons', 'circles', 'linear'])
    parser.add_argument('--noncvx', action='store_true')

    args = parser.parse_args()

    npr.seed(args.seed)
    tf.set_random_seed(args.seed)

    setproctitle.setproctitle('bamos.icnn.synthetic.{}.{}'.format(args.model, args.dataset))

    save = os.path.join(os.path.expanduser(args.save),
                        "{}.{}".format(args.model, args.dataset))
    if os.path.isdir(save):
        shutil.rmtree(save)
    os.makedirs(save, exist_ok=True)

    if args.dataset == "moons":
        (dataX, dataY) = make_moons(noise=0.3, random_state=0)
    elif args.dataset == "circles":
        (dataX, dataY) = make_circles(noise=0.2, factor=0.5, random_state=0)
        dataY = 1.-dataY
    elif args.dataset == "linear":
        (dataX, dataY) = make_classification(n_features=2, n_redundant=0, n_informative=2,
                                             random_state=1, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        dataX += 2 * rng.uniform(size=dataX.shape)
    else:
        assert(False)

    dataY = dataY.reshape((-1, 1)).astype(np.float32)

    nData = dataX.shape[0]
    nFeatures = dataX.shape[1]
    nLabels = 1
    nXy = nFeatures + nLabels

    config = tf.ConfigProto() #log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = Model(nFeatures, nLabels, sess, args.model, nGdIter=30)
        model.train(args, dataX, dataY)

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
    def __init__(self, nFeatures, nLabels, sess, model, nGdIter):
        self.nFeatures = nFeatures
        self.nLabels = nLabels
        self.sess = sess
        self.model = model

        self.trueY_ = tf.placeholder(tf.float32, shape=[None, nLabels], name='trueY')

        self.x_ = tf.placeholder(tf.float32, shape=[None, nFeatures], name='x')
        self.y0_ = tf.placeholder(tf.float32, shape=[None, nLabels], name='y')

        if model == 'picnn':
            f = self.f_picnn
        elif model == 'ficnn':
            f = self.f_ficnn

        E0_ = f(self.x_, self.y0_)

        lr = 0.01
        momentum = 0.9

        yi_ = self.y0_
        Ei_ = E0_
        vi_ = 0

        for i in range(nGdIter):
            prev_vi_ = vi_
            vi_ = momentum*prev_vi_ - lr*tf.gradients(Ei_, yi_)[0]
            yi_ = yi_ - momentum*prev_vi_ + (1.+momentum)*vi_
            Ei_ = f(self.x_, yi_, True)

        self.yn_ = yi_
        self.energies_ = Ei_

        self.mse_ = tf.reduce_mean(tf.square(self.yn_ - self.trueY_))

        self.opt = tf.train.AdamOptimizer(0.001)
        self.theta_ = tf.trainable_variables()
        self.gv_ = [(g,v) for g,v in
                    self.opt.compute_gradients(self.mse_, self.theta_)
                    if g is not None]
        self.train_step = self.opt.apply_gradients(self.gv_)

        self.theta_cvx_ = [v for v in self.theta_
                           if 'proj' in v.name and 'W:' in v.name]

        self.makeCvx = [v.assign(tf.abs(v)/10.) for v in self.theta_cvx_]
        self.proj = [v.assign(tf.maximum(v, 0)) for v in self.theta_cvx_]

        # for g,v in self.gv_:
        #     variable_summaries(g, 'gradients/'+v.name)

        self.merged = tf.merge_all_summaries()
        self.saver = tf.train.Saver(max_to_keep=0)

    def train(self, args, dataX, dataY):
        save = os.path.join(os.path.expanduser(args.save),
                            "{}.{}".format(args.model, args.dataset))

        nTrain = dataX.shape[0]

        imgDir = os.path.join(save, 'imgs')
        if not os.path.exists(imgDir):
            os.makedirs(imgDir)

        trainFields = ['iter', 'loss']
        trainF = open(os.path.join(save, 'train.csv'), 'w')
        trainW = csv.writer(trainF)
        trainW.writerow(trainFields)

        self.trainWriter = tf.train.SummaryWriter(os.path.join(save, 'train'),
                                                  self.sess.graph)
        self.sess.run(tf.initialize_all_variables())
        if not args.noncvx:
            self.sess.run(self.makeCvx)

        nParams = np.sum(v.get_shape().num_elements() for v in tf.trainable_variables())

        meta = {'nTrain': nTrain, 'nParams': nParams, 'nEpoch': args.nEpoch}
        metaP = os.path.join(save, 'meta.json')
        with open(metaP, 'w') as f:
            json.dump(meta, f, indent=2)

        bestMSE = None
        for i in range(args.nEpoch):
            tflearn.is_training(True)

            print("=== Epoch {} ===".format(i))
            start = time.time()

            y0 = np.full(dataY.shape, 0.5)
            _, trainMSE, yn = self.sess.run(
                [self.train_step, self.mse_, self.yn_],
                feed_dict={self.x_: dataX, self.y0_: y0, self.trueY_: dataY})
            if not args.noncvx and len(self.proj) > 0:
                self.sess.run(self.proj)

            trainW.writerow((i, trainMSE))
            trainF.flush()

            print(" + loss: {:0.5e}".format(trainMSE))
            print(" + time: {:0.2f} s".format(time.time()-start))

            if i % 10 == 0:
                loc = "{}/{:05d}".format(imgDir, i)
                self.plot(loc, dataX, dataY)

            if bestMSE is None or trainMSE < bestMSE:
                loc = os.path.join(save, 'best')
                self.plot(loc, dataX, dataY)
                bestMSE = trainMSE

        trainF.close()

    def f_ficnn(self, x, y, reuse=False):
        fc = tflearn.fully_connected
        xy = tf.concat(1, (x, y))

        prevZ = None
        for i, sz in enumerate([200, 200, 1]):
            z_add = []

            with tf.variable_scope('z_x{}'.format(i)) as s:
                z_x = fc(xy, sz, reuse=reuse, scope=s, bias=True)
                z_add.append(z_x)

            if prevZ is not None:
                with tf.variable_scope('z_z{}_proj'.format(i)) as s:
                    z_z = fc(prevZ, sz, reuse=reuse, scope=s, bias=False)
                    z_add.append(z_z)

            if sz != 1:
                z = tf.nn.relu(tf.add_n(z_add))
            prevZ = z

        return tf.contrib.layers.flatten(z)

    def f_picnn(self, x, y, reuse=False):
        fc = tflearn.fully_connected
        xy = tf.concat(1, (x, y))

        prevZ, prevU = None, x
        for layerI, sz in enumerate([200, 200, 1]):
            if sz != 1:
                with tf.variable_scope('u'+str(layerI)) as s:
                    u = fc(prevU, sz, scope=s, reuse=reuse)
                    u = tf.nn.relu(u)

            z_add = []

            if prevZ is not None:
                with tf.variable_scope('z{}_zu_u'.format(layerI)) as s:
                    prevU_sz = prevU.get_shape()[1].value
                    zu_u = fc(prevU, prevU_sz, reuse=reuse, scope=s,
                            activation='relu', bias=True)
                with tf.variable_scope('z{}_zu_proj'.format(layerI)) as s:
                    z_zu = fc(tf.mul(prevZ, zu_u), sz, reuse=reuse, scope=s,
                                bias=False)
                z_add.append(z_zu)

            with tf.variable_scope('z{}_yu_u'.format(layerI)) as s:
                yu_u = fc(prevU, self.nLabels, reuse=reuse, scope=s, bias=True)
            with tf.variable_scope('z{}_yu'.format(layerI)) as s:
                z_yu = fc(tf.mul(y, yu_u), sz, reuse=reuse, scope=s, bias=False)
            z_add.append(z_yu)

            with tf.variable_scope('z{}_u'.format(layerI)) as s:
                z_u = fc(prevU, sz, reuse=reuse, scope=s, bias=True)
            z_add.append(z_u)

            z = tf.add_n(z_add)
            if sz != 1:
                z = tf.nn.relu(z)

            prevU = u
            prevZ = z

        return tf.contrib.layers.flatten(z)

    def plot(self, loc, dataX, dataY):
        delta = 0.01
        x_min, x_max = dataX[:, 0].min() - .5, dataX[:, 0].max() + .5
        y_min, y_max = dataX[:, 1].min() - .5, dataX[:, 1].max() + .5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 20),
                             np.linspace(y_min, y_max, 20))
        xxFlat = xx.ravel()
        yyFlat = yy.ravel()
        gridX = np.vstack((xxFlat, yyFlat)).T
        y0 = np.full(gridX.shape[0], 0.5).reshape((-1, 1))
        yn, = self.sess.run([self.yn_], feed_dict={self.x_: gridX, self.y0_: y0})
        yn = np.clip(yn, 0., 1.)
        zz = 1.-yn.reshape(xx.shape)

        fig, ax = plt.subplots(1, 1, figsize=(5,5))
        plt.axis([x_min, x_max, y_min, y_max])
        fig.tight_layout()
        fig.subplots_adjust(bottom=0,top=1,left=0,right=1)
        ax.set_autoscale_on(False)
        ax.grid(False)
        v = np.linspace(0.0, 1.0, 10, endpoint=True)
        plt.contourf(xx, yy, zz, v, alpha=0.5, cmap=cm.bwr)
        # plt.colorbar()
        yFlat = dataY.ravel()
        plt.scatter(dataX[yFlat == 0, 0], dataX[yFlat == 0, 1], color='red')
        plt.scatter(dataX[yFlat == 1, 0], dataX[yFlat == 1, 1], color='blue')
        for ext in ['png', 'pdf']:
            plt.savefig('{}.{}'.format(loc, ext))
        plt.close()


if __name__=='__main__':
    main()
