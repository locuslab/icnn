#!/usr/bin/env python3

import argparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')
import numpy as np
import pandas as pd
import math

import os
import sys
import json
import glob

scriptDir = os.path.dirname(os.path.realpath(__file__))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('workDir', type=str)
    # parser.add_argument('--ymin', type=float, default=1e-4)
    # parser.add_argument('--ymax', type=float, default=1e-1)
    args = parser.parse_args()

    trainF = os.path.join(args.workDir, 'train.csv')
    if os.path.isfile(trainF):
        trainDf, testDf, meta = getDataSingle(args.workDir)
    else:
        assert(False)
        # data = getDataMulti(args.workDir)

    # plotF1(trainDf, testDf, meta, args.workDir)
    plotLoss(trainDf, testDf, meta, args.workDir)

def getDataSingle(workDir):
    trainF = os.path.join(workDir, 'train.csv')
    testF = os.path.join(workDir, 'test.csv')
    metaF = os.path.join(workDir, 'meta.json')

    trainDf = pd.read_csv(trainF, sep=',')
    testDf = pd.read_csv(testF, sep=',')

    # trainDf = pd.read_csv(trainF, sep=',', header=None,
    #                       names=['iter', 'f1', 'loss'])
    # testDf = pd.read_csv(testF, sep=',', header=None,
    #                       names=['iter', 'f1'])

    with open(metaF, 'r') as f:
        meta = json.load(f)

    return trainDf, testDf, meta

def getDataMulti(workDir):
    assert(False)
    ds = list(glob.glob("{}/*/".format(workDir)))

    trainIters = []
    # trainF1s = []
    trainLoss = []
    testIters = []
    # testF1s = []

    lastTrainIter = None

    for d in sorted(ds):
        trainDf = pd.read_csv(os.path.join(d, 'train.csv'), sep=',',
                              header=None, names=['iter', 'loss'])
        testDf = pd.read_csv(os.path.join(d, 'test.csv'), sep=',',
                             header=None, names=['iter', 'loss'])
        trainIter = trainDf['iter'].values
        testIter = testDf['iter'].values
        if lastTrainIter is not None:
            trainIter += lastTrainIter
            testIter += lastTrainIter
        trainIters.append(trainIter)
        testIters.append(testIter)
        # trainF1s.append(trainDf['f1'].values)
        trainLoss.append(trainDf['loss'].values)
        # testF1s.append(testDf['f1'].values)
        lastTrainIter = trainIter[-1]
    trainIters = np.concatenate(trainIters)
    # trainF1s = np.concatenate(trainF1s)
    trainLoss = np.concatenate(trainLoss)
    testIters = np.concatenate(testIters)
    # testF1s = np.concatenate(testF1s)

    metaF = ds[0] + '/opt.json'

    with open(metaF, 'r') as f:
        meta = json.load(f)

    return [trainIters, trainLoss, testIters, meta]

def plotF1(trainDf, testDf, meta, workDir):
    nTrain = meta['nTrain']
    trainBatchSz = meta['trainBatchSz']

    # fig, ax = plt.subplots(1, 1, figsize=(5,2))
    fig, ax = plt.subplots(1, 1)
    fig.tight_layout()
    # fig.subplots_adjust(bottom=0.25,left=0.15) # For (5, 2)
    fig.subplots_adjust(bottom=0.1,left=0.1)
    N = math.ceil(nTrain/trainBatchSz)

    trainIters = trainDf['iter'].values
    trainF1s = trainDf['f1'].values

    trainIters = trainIters[N:]*trainBatchSz/nTrain
    trainF1s = [sum(trainF1s[i-N:i])/N for i in range(N, len(trainF1s))]
    plt.plot(trainIters, trainF1s, label='Train')

    testIters = testDf['iter'].values
    testF1s = testDf['f1'].values
    if len(testF1s) > 0:
        plt.plot(testIters*trainBatchSz/nTrain, testF1s, label='Test')
    # trainP = testDf['f1'].plot(ax=ax)
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.ylim(ymin=0, ymax=1)
    plt.xlim(xmin=0)
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)
    # ax.set_yscale('log')
    plt.legend()
    for ext in ['pdf', 'png']:
        f = os.path.join(workDir, "f1s."+ext)
        fig.savefig(f)
        print("Created {}".format(f))

def plotLoss(trainDf, testDf, meta, workDir):
    nTrain = meta['nTrain']
    trainBatchSz = meta['trainBatchSz']

    # fig, ax = plt.subplots(1, 1, figsize=(5,2))
    fig, ax = plt.subplots(1, 1)
    fig.tight_layout()
    # fig.subplots_adjust(bottom=0.25,left=0.15) # For (5, 2)
    fig.subplots_adjust(bottom=0.1,left=0.1)
    N = math.ceil((10.0/10.0)*nTrain/meta['trainBatchSz'])

    trainIters = trainDf['iter'].values
    trainLoss = trainDf['loss'].values

    trainIters = trainIters[N:]*trainBatchSz/nTrain
    trainLoss = [sum(trainLoss[i-N:i])/N for i in range(N, len(trainLoss))]
    plt.plot(trainIters, trainLoss, label='Train')

    if 'loss' in testDf:
        testIters = testDf['iter'].values
        testLoss = testDf['loss'].values
        if len(testLoss) > 0:
            plt.plot(testIters*trainBatchSz/nTrain, testLoss, label='Test')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)
    plt.legend()
    ax.set_yscale('log')
    # plt.legend()
    for ext in ['pdf', 'png']:
        f = os.path.join(workDir, "loss."+ext)
        fig.savefig(f)
        print("Created {}".format(f))


if __name__ == '__main__':
    main()
