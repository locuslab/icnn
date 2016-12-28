#!/usr/bin/env python3

import argparse
import os
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('expDir', type=str)
    parser.add_argument('--xmax', type=float)
    parser.add_argument('--ymin', type=float, default=0.0)
    parser.add_argument('--ymax', type=float)
    args = parser.parse_args()

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plt.xlabel('Timestep')
    plt.ylabel('Reward')

    trainP = os.path.join(args.expDir, 'train.log')
    trainData = np.loadtxt(trainP).reshape(-1, 2)
    testP = os.path.join(args.expDir, 'test.log')
    testData = np.loadtxt(testP).reshape(-1, 2)
    if trainData.shape[0] > 1:
        plt.plot(trainData[:,0], trainData[:,1], label='Train')
    if testData.shape[0] > 1:
        testI = testData[:,0]
        testRew = testData[:,1]
        plt.plot(testI, testRew, label='Test')

        N = 10
        testI_ = testI[N:]
        testRew_ = [sum(testRew[i-N:i])/N for i in range(N, len(testRew))]
        plt.plot(testI_, testRew_, label='Rolling Test')

    plt.ylim([args.ymin, args.ymax])
    plt.legend()
    fname = os.path.join(args.expDir, 'reward.pdf')
    plt.savefig(fname)
    print('Created {}'.format(fname))

if __name__ == '__main__':
    main()
