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
    parser.add_argument('data', type=str, help='dir contains outputs of DDPG, NAF and ICNN')
    parser.add_argument('--xmax', type=float)
    parser.add_argument('--ymin', type=float, default=0.0)
    parser.add_argument('--ymax', type=float)
    args = parser.parse_args()

    names = ['DDPG', 'NAF', 'ICNN']
    colors = ['red', 'blue', 'green']

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plt.xlabel('Timestep')
    plt.ylabel('Reward')

    lines = []
    for name, color in zip(names, colors):
        dirpath = os.path.join(args.data, name)
        if os.path.isdir(dirpath):
            X, Ymin, Ymax, Ymean = get_data(dirpath)
            line, = plt.plot(X, Ymean, label=name, color=color)
            lines.append(line)
            plt.fill_between(X, Ymin, Ymax, alpha=0.1, color=color)
    plt.ylim([args.ymin, args.ymax])
    plt.legend(handles=lines, loc=2)
    fname = os.path.join(args.data, 'result.pdf')
    plt.savefig(fname)
    print('Created {}'.format(fname))


def get_data(dirpath):
    minX, maxX = None, None
    X, Y = [], []
    for d in os.listdir(dirpath):
        logName = os.path.join(dirpath, d, 'log.txt')
        if not os.path.exists(logName):
            logName = os.path.join(dirpath, d, 'test.log')
            if not os.path.exists(logName):
                print("Log file not found for: {}".format(os.path.join(dirpath, d)))
                continue

        data = np.loadtxt(logName)
        x = data[:,0].ravel()
        y = data[:,1].ravel()

        minX = np.min(x) if not minX else min(np.min(x), minX)
        maxX = np.max(x) if not maxX else max(np.max(x), maxX)
        X.append(x)
        Y.append(y)

    xs = np.linspace(minX, maxX, 10000)

    interpY = []
    for x,y in zip(X,Y):
        interpY.append(np.interp(xs, x, y))

    Y = np.asarray(interpY)
    Ysdom = 1.96 * np.std(Y, axis=0) / np.sqrt(Y.shape[0])
    Ymean = np.mean(Y, axis=0)

    return xs, Ymean - Ysdom, Ymean + Ysdom, Ymean


if __name__ == '__main__':
    main()
