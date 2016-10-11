#!/usr/bin/env python3.4

import argparse

import numpy as np
import numpy.random as npr

np.set_printoptions(precision=2)

import os
import sys
import time

import tensorflow as tf
import tflearn
import tflearn.initializations as tfi
import tflearn.data_flow
import tflearn.helpers as tfh

from datetime import datetime

import cvxpy as cp
import pandas as pd

import csv
import json
import pickle as pkl

import setproctitle

import bibsonomy
import util

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='work-ff')
    parser.add_argument('--nEpoch', type=float, default=100)
    parser.add_argument('--trainBatchSz', type=int, default=128)
    parser.add_argument('--testBatchSz', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--layerSizes', type=int, nargs='+', default=[600])
    # parser.add_argument('--saveFeatures', action='store_true')
    parser.add_argument('--dataset', type=str, choices=['bibtex', 'bookmarks', 'delicious'],
                        default='bibtex')
    parser.add_argument('--valSplit', type=float, default=0)

    args = parser.parse_args()
    assert(args.valSplit < 1.)

    setproctitle.setproctitle('bamos.ff.{}.{}'.format(
        args.dataset,
        ','.join(str(x) for x in args.layerSizes)))

    npr.seed(args.seed)
    tf.set_random_seed(args.seed)

    save = os.path.expanduser(args.save)
    if not os.path.isdir(save):
        os.makedirs(save, exist_ok=True)

    if args.dataset == 'bibtex':
        data = bibsonomy.loadBibtex("data/bibtex")
    elif args.dataset == 'bookmarks':
        data = bibsonomy.loadBookmarks("data/bookmarks")
    elif args.dataset == 'delicious':
        data = bibsonomy.loadDelicious("data/delicious")
    else:
        assert(False)

    nFeatures = data['trainX'].shape[1]
    nLabels = data['trainY'].shape[1]
    nXy = nFeatures + nLabels

    x_ = tf.placeholder(tf.float32, shape=(None, nFeatures), name='x')
    y_ = tf.placeholder(tf.float32, shape=(None, nLabels), name='y')
    net = x_

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
        testDf = tflearn.data_flow.FeedDictFlow(
            {x_: valX, y_: valY},
            tf.train.Coordinator(),
            batch_size=args.testBatchSz)
    else:
        trainX = data['trainX']
        trainY = data['trainY']
        testDf = tflearn.data_flow.FeedDictFlow(
            {x_: data['testX'], y_: data['testY']},
            tf.train.Coordinator(),
            batch_size=args.testBatchSz)


    # nTrain = data['trainX'].shape[0]
    nTest = data['testX'].shape[0]

    print("\n\n" + "="*40)
    print("+ nTrain: {}, nVal: {}, nTest: {}".format(nTrain, nVal, nTest))
    print("+ nFeatures: {}, nLabels: {}".format(nFeatures, nLabels))
    print("="*40 + "\n\n")

    nIter = int(args.nEpoch*np.ceil(nTrain/args.trainBatchSz))

    with tf.variable_scope("FeedForward") as scope:
        for sz in args.layerSizes:
            std = 1.0/np.sqrt(sz)
            net = tflearn.fully_connected(net, sz, activation='relu',
                                        weight_decay=0,
                                        weights_init=tfi.uniform(None, -std, std),
                                        bias_init=tfi.uniform(None, -std, std))
            net = tflearn.layers.normalization.batch_normalization(net)

        features_ = net
        sz = nLabels
        std = 1.0/np.sqrt(sz)
        logits_ = tflearn.fully_connected(net, sz, activation='linear', weight_decay=0,
                                        weights_init=tfi.uniform(None, -std, std),
                                        bias_init=tfi.uniform(None, -std, std))
        yhat_ = tf.sigmoid(logits_)

    ff_vars = tf.all_variables()

    # loss_ = tf.reduce_mean(tf.square(y_ - yhat_))
    loss_ = -tf.reduce_sum(y_*tf.log(yhat_+1e-10)) \
            -tf.reduce_sum((1.-y_)*tf.log(1.-yhat_+1e-10))

    train_step = tf.train.AdamOptimizer(0.001).minimize(loss_)

    trainFields = ['iter', 'f1', 'loss']
    trainF = open(os.path.join(save, 'train.csv'), 'w')
    trainW = csv.writer(trainF)
    trainW.writerow(trainFields)

    testFields = ['iter', 'f1', 'loss']
    testF = open(os.path.join(save, 'test.csv'), 'w')
    testW = csv.writer(testF)
    testW.writerow(testFields)

    meta = {'nEpoch': args.nEpoch, 'nIter': nIter,
            'nTrain': nTrain, 'nVal': nVal, 'nTest': nTest,
            'trainBatchSz': args.trainBatchSz,
            'layerSizes': args.layerSizes}

    ff_saver = tf.train.Saver(ff_vars, max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        graphWriter = tf.train.SummaryWriter(os.path.join(save, 'graph'), sess.graph)
        sess.run(tf.initialize_all_variables())

        nParams = np.sum(v.get_shape().num_elements() for v in tf.trainable_variables())
        meta['nParams'] = nParams
        with open(os.path.join(args.save, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)


        bestTestF1 = 0
        for i in range(nIter):
            tflearn.is_training(True)

            I = npr.randint(nTrain, size=args.trainBatchSz)
            xBatch = trainX[I, :]
            yBatch = trainY[I, :]

            _, loss, yPred = sess.run(
                [train_step, loss_, yhat_],
                feed_dict={x_: xBatch, y_: yBatch})
            trainF1 = util.macroF1(yBatch, yPred)
            trainW.writerow((i, trainF1, loss))
            trainF.flush()

            if i % np.ceil((nTrain/1.0)/args.trainBatchSz) == 0:
                print("=== Iteration {} (Epoch {:.2f}) ===".format(
                    i, i/np.ceil(nTrain/args.trainBatchSz)))
                print(" + train F1: {:0.4f}".format(trainF1))
                print(" + train loss: {:0.2e}".format(loss))

            if i % np.ceil(nTrain/args.trainBatchSz) == 0:
                tflearn.is_training(False)
                # testF1, testLoss = tfh.trainer.evaluate_flow(sess, [F1_, loss_], testDf)[0]
                yPred, testLoss = sess.run([yhat_, loss_],
                                           feed_dict={x_: data['testX'], y_: data['testY']})
                testF1 = util.macroF1(data['testY'], yPred)
                print(" + testF1: {:0.4f}".format(testF1))
                print(" + testLoss: {:0.2e}".format(testLoss))
                testW.writerow((i, testF1, testLoss))
                testF.flush()

                # os.system('./icnn.plot.py ' + args.save)

                if testF1 > bestTestF1:
                    bestTestF1 = testF1
                    # print("  + Saving new best model.")
                    # ff_saver.save(sess, os.path.join(args.save, 'best.tf'))

                    # trainL = sess.run(logits_, feed_dict={x_: data['trainX']})
                    # testL = sess.run(logits_, feed_dict={x_: data['testX']})
                    # p = os.path.join(args.save, 'best.logits.pkl')
                    # print("  + Writing logits to: ", p)
                    # featuresData = {'trainX': trainL, 'trainY': data['trainY'],
                    #                 'testX': testL, 'testY': data['testY']}
                    # with open(p, 'wb') as f:
                    #     pkl.dump(featuresData, f)


    trainF.close()
    testF.close()

    with open(os.path.join(args.save, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    os.system('./icnn.plot.py ' + args.save)

if __name__ == '__main__':
    main()
