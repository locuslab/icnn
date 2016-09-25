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

import setproctitle

from datetime import datetime

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

sys.path.append('../lib')

import bibsonomy
import bundle_entropy
import icnn_ebundle
import bamos_opt

def entr(x):
    z = -x * np.log(x) - (1.-x)*np.log(1.-x)
    z[z!=z] = 0.0
    return np.sum(z, axis=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('chkpt', type=str)
    parser.add_argument('--save', type=str, default='work')
    parser.add_argument('--layerSizes', type=int, nargs='+', default=[600, 600])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, choices=['bibtex', 'bookmarks', 'delicious'],
                        default='bibtex')

    args = parser.parse_args()

    setproctitle.setproctitle('bamos.icnn.ebundle-vs-gd.{}.{}'.format(
        args.dataset,
        ','.join(str(x) for x in args.layerSizes)))

    npr.seed(args.seed)
    tf.set_random_seed(args.seed)

    if args.dataset == 'bibtex':
        data = bibsonomy.loadBibtex("data/bibtex")
    elif args.dataset == 'bookmarks':
        data = bibsonomy.loadBookmarks("data/bookmarks")
    elif args.dataset == 'delicious':
        data = bibsonomy.loadDelicious("data/delicious")
    else:
        assert(False)

    nTest = data['testX'].shape[0]
    nFeatures = data['trainX'].shape[1]
    nLabels = data['trainY'].shape[1]
    nXy = nFeatures + nLabels

    nTrain = data['trainX'].shape[0]
    trainX = data['trainX']
    trainY = data['trainY']

    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = icnn_ebundle.Model(nFeatures, nLabels, args.layerSizes, sess)
        model.load(args.chkpt)

        nSamples = 10

        # Bundle Entropy
        bundleIter, bundleTime, bundleEs = [], [], []
        def fg(yhats):
            fd = {model.x_: xBatch, model.y_: yhats}
            e, ge = model.sess.run([model.E_, model.dE_dy_], feed_dict=fd)
            return e, ge

        def cb(iterNum, es, x):
            bundleIter.append(iterNum)
            print(np.mean(es))
            es_entr = es - entr(x)
            bundleEs.append(np.mean(es_entr))
            bundleTime.append(time.time()-start)

        start = time.time()
        I = npr.randint(nTrain, size=nSamples)
        xBatch = trainX[I, :]
        yBatch = trainY[I, :]
        y0 = np.full(yBatch.shape, 0.5)
        yN, G, h, lam, ys, nIters = bundle_entropy.solveBatch(
            fg, y0, nIter=10, callback=cb)

        # PGD
        pgdIter, pgdTime, pgdEs = {}, {}, {}
        def fg(yhats):
            fd = {model.x_: xBatch, model.y_: yhats}
            e, ge = model.sess.run([model.E_entr_, model.dE_entr_dy_], feed_dict=fd)
            return e, ge

        def proj(x):
            return np.clip(x, 1e-6, 1.-1e-6)

        lrs = [0.1, 0.01, 0.001]
        for lr in lrs:
            pgdIter[lr] = []
            pgdTime[lr] = []
            pgdEs[lr] = []
            def cb(iterNum, es, gs, bestM):
                pgdIter[lr].append(iterNum)
                pgdEs[lr].append(np.mean(es))
                pgdTime[lr].append(time.time()-start)

            start = time.time()
            y0 = np.full(yBatch.shape, 0.5)
            bamos_opt.pgd.solve_batch(fg, proj, y0, lr=lr, rollingDecay=0.5, eps=1e-3,
                                      minIter=10, maxIter=10, callback=cb)

        fig, ax = plt.subplots(1, 1)
        plt.xlabel('Iteration')
        plt.ylabel('Entropy-Scaled Objective')
        for lr in lrs:
            plt.plot(pgdIter[lr], pgdEs[lr], label='PGD, lr={}'.format(lr))
        plt.plot(bundleIter, bundleEs, label='Bundle Entropy', color='k',
                 linestyle='dashed')
        plt.legend()
        ax.set_yscale('log')
        for ext in ['png', 'pdf']:
            fname = os.path.join(args.save, 'obj.'+ext)
            plt.savefig(fname)
            print("Created {}".format(fname))


if __name__=='__main__':
    main()
