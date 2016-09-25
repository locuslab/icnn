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

parser = argparse.ArgumentParser()
parser.add_argument('--save', type=str, default='work/icnn')
parser.add_argument('--nEpoch', type=int, default=100)
parser.add_argument('--trainBatchSz', type=int, default=128)
# parser.add_argument('--testBatchSz', type=int, default=2048)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--data', type=str)

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

nTrain = data['trainX'].shape[0]
nTest = data['testX'].shape[0]

nFeatures = data['trainX'].shape[1]
nLabels = data['trainY'].shape[1]
nXy = nFeatures + nLabels

print("\n\n" + "="*40)
print("+ nTrain: {}, nTest: {}".format(nTrain, nTest))
print("+ nFeatures: {}, nLabels: {}".format(nFeatures, nLabels))
print("="*40 + "\n\n")

nIter = int(np.ceil(args.nEpoch*nTrain/args.trainBatchSz))

def tf_initW(inSz, outSz):
    stdev = 1./np.sqrt(outSz)
    return tf.Variable(tf.random_uniform([inSz, outSz], -stdev, stdev))

def tf_initW_pos(inSz, outSz):
    stdev = 1./np.sqrt(outSz)
    return tf.Variable(tf.random_uniform([inSz, outSz], 0, stdev))

def tf_initB(outSz):
    return tf.Variable(tf.zeros([outSz]))

szs = [600, 600]
assert(len(szs) == 2)
Ws = {
    'z1_xy': tf_initW(nXy, szs[0]),

    'z2_u': tf_initW(szs[0], szs[1]),
    'z2_y': tf_initW(nLabels, szs[1]),
    'z2_z': tf_initW_pos(szs[0], szs[1]),
    'z2_uz': tf_initW_pos(szs[0], szs[1]),

    'z3_u': tf_initW(szs[1], 1),
    'z3_y': tf_initW(nLabels, 1),
    'z3_z': tf_initW_pos(szs[1], 1),
    'z3_uz': tf_initW_pos(szs[1], 1),

    'u1': tf_initW(nFeatures, szs[0]),
    'u2': tf_initW(szs[0], szs[1]),
}
bs = {
    'z1': tf_initB(szs[0]),
    'z2': tf_initB(szs[1]),
    'z3': tf_initB(1),

    'u1': tf_initB(szs[0]),
    'u2': tf_initB(szs[1]),
}

proj = [
    Ws['z2_z'].assign(tf.maximum(Ws['z2_z'], 0)),
    Ws['z2_uz'].assign(tf.maximum(Ws['z2_uz'], 0)),
    Ws['z3_z'].assign(tf.maximum(Ws['z3_z'], 0)),
    Ws['z3_uz'].assign(tf.maximum(Ws['z3_uz'], 0)),
]

def f(x, y, reuse=False):
    act = tf.nn.relu
    # act = tf.nn.softplus
    xy = tf.concat(1, (x, y))
    z1 = act(tf.matmul(xy, Ws['z1_xy']) + bs['z1'], name='z1')

    u1 = act(tf.matmul(x, Ws['u1']) + bs['u1'], name='u1')
    with tf.variable_scope("u1_bn") as s:
        u1 = tflearn.layers.normalization.batch_normalization(u1, reuse=reuse, scope=s)

    z2_z = tf.matmul(z1, Ws['z2_z'])
    z2_u = tf.matmul(u1, Ws['z2_u'])
    z2_y = tf.matmul(y, Ws['z2_y'])
    uz1 = tf.mul(u1, z1)
    z2_uz = tf.matmul(uz1, Ws['z2_uz'])
    z2 = act(z2_z + z2_u + z2_y + z2_uz + bs['z2'], name='z2')

    u2 = act(tf.matmul(u1, Ws['u2']) + bs['u2'], name='u2')
    with tf.variable_scope("u2_bn") as s:
        u2 = tflearn.layers.normalization.batch_normalization(u2, reuse=reuse, scope=s)

    z3_z = tf.matmul(z2, Ws['z3_z'])
    z3_u = tf.matmul(u2, Ws['z3_u'])
    z3_y = tf.matmul(y, Ws['z3_y'])
    uz2 = tf.mul(u2, z2)
    z3_uz = tf.matmul(uz2, Ws['z3_uz'])
    z3 = z3_z + z3_u + z3_y + z3_uz + bs['z3']
    z3 = tf.identity(z3, name='z3')

    return tf.reshape(z3, [-1])

lr = 0.01
momentum = 0.9
nPgdIter = 10

trueY_ = tf.placeholder(tf.float32, shape=(None, nLabels), name='trueY')

x_ = tf.placeholder(tf.float32, shape=(None, nFeatures), name='x')
y0_ = tf.placeholder(tf.float32, shape=(None, nLabels), name='y')
E0_ = f(x_, y0_)

yi_ = y0_
Ei_ = E0_
vi_ = 0

for i in range(nPgdIter):
    prev_vi_ = vi_
    vi_ = momentum*prev_vi_ - lr*tf.gradients(Ei_, yi_)[0]
    yi_ = yi_ - momentum*prev_vi_ + (1.+momentum)*vi_
    Ei_ = f(x_, yi_, True)

yn_ = yi_
energies_ = Ei_

mse_ = tf.reduce_mean(tf.square(yn_ - trueY_))

opt = tf.train.AdamOptimizer(0.001)
theta_ = tf.trainable_variables()
gv_ = [(g,v) for g,v in opt.compute_gradients(mse_, theta_) if g is not None]
train_step = opt.apply_gradients(gv_)

def tf_nOnes(b):
    # Must be binary.
    return tf.reduce_sum(tf.cast(b, tf.int32))

yBinary_ = tf.select(yn_ < 0.5, tf.zeros_like(yn_), tf.ones_like(yn_))
F1_ = tf.reduce_mean(2*tf_nOnes(tf.mul(trueY_, yBinary_))/
                     (tf_nOnes(trueY_) + tf_nOnes(yBinary_)))

trainFields = ['iter', 'f1', 'loss']
trainF = open(os.path.join(save, 'train.csv'), 'w')
trainW = csv.writer(trainF)
trainW.writerow(trainFields)

testFields = ['iter', 'f1', 'loss']
testF = open(os.path.join(save, 'test.csv'), 'w')
testW = csv.writer(testF)
testW.writerow(testFields)

config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.initialize_all_variables())
    graphWriter = tf.train.SummaryWriter(os.path.join(save, 'graph'), sess.graph)

    nParams = np.sum(v.get_shape().num_elements() for v in tf.trainable_variables())

    meta = {'nTrain': nTrain, 'trainBatchSz': args.trainBatchSz,
            'nParams': nParams, 'nEpoch': args.nEpoch,
            'nIter': nIter}
    metaP = os.path.join(save, 'meta.json')
    with open(metaP, 'w') as f:
        json.dump(meta, f, indent=2)

    for i in range(nIter):
        tflearn.is_training(True)

        print("=== Iteration {} (Epoch {:.2f}) ===".format(
            i, i/np.ceil(nTrain/args.trainBatchSz)))
        I = npr.randint(nTrain, size=args.trainBatchSz)
        xBatch = data['trainX'][I, :]
        yBatch = data['trainY'][I, :]


        y0 = np.full(yBatch.shape, 0.5)
        _, trainF1, trainMSE, yn = sess.run(
            [train_step, F1_, mse_, yn_],
            feed_dict={x_: xBatch, y0_: y0, trueY_: yBatch})
        sess.run(proj)

        trainW.writerow((i, trainF1, trainMSE))
        trainF.flush()

        print(" + trainF1: {:0.2f}".format(trainF1))
        print(" + MSE: {:0.2e}".format(trainMSE))

        if (i+1) % np.ceil(nTrain/args.trainBatchSz) == 0:
            print("=== Testing ===")
            tflearn.is_training(False)
            y0 = np.full(data['testY'].shape, 0.5)
            testF1 = sess.run(F1_, feed_dict=
                {x_: data['testX'], trueY_: data['testY'], y0_: y0})
            print(" + testF1: {:0.4f}".format(testF1))
            testW.writerow((i, testF1))
            testF.flush()

trainF.close()
testF.close()

os.system('./icnn.plot.py ' + args.save)
