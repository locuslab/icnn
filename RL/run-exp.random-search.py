#!/usr/bin/env python3

import argparse
import os
import sys
import shutil
from subprocess import Popen, PIPE

import numpy as np
import numpy.random as npr

pythonCmd = 'python3'
rlDir = os.path.dirname(os.path.realpath(__file__))
plotSrc = os.path.join(rlDir, 'plot-all.py')
mainSrc = os.path.join(rlDir, 'src', 'main.py')

all_algs = ['DDPG', 'NAF', 'ICNN']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str,
                        choices=['InvertedPendulum', 'InvertedDoublePendulum',
                                 'Reacher', 'HalfCheetah', 'Swimmer', 'Hopper',
                                 'Walker2d', 'Ant', 'Humanoid', 'HumanoidStandup'],
                        help='(Every task is currently v1.)')
    parser.add_argument('--alg', type=str, choices=all_algs)
    parser.add_argument('--nSamples', type=int, default=50)
    parser.add_argument('--save', type=str)
    parser.add_argument('--overwrite', action='store_true')

    args = parser.parse_args()

    allDir = args.save or os.path.join('output.random-search', args.task)
    if os.path.exists(allDir):
        if args.overwrite:
            shutil.rmtree(allDir)
    os.makedirs(allDir, exist_ok=True)

    algs = [args.alg] if args.alg is not None else all_algs
    np.random.seed(0)
    for i in range(args.nSamples):
        l1size = npr.randint(100, 600)
        l2size = npr.randint(100, l1size)
        hp_alg = {
            'l1size': l1size,
            'l2size': l2size,
            'reward_k': 10.**npr.uniform(-4, 1),
            'l2norm': 10.**npr.uniform(-10, -2),
            'pl2norm': 10.**npr.uniform(-10, -2),
            'rate': 10.**npr.uniform(-4, -1),
            'prate': 10.**npr.uniform(-4, -1),
            'outheta': np.maximum(1e-8, npr.normal(loc=0.15, scale=0.1)),
            'ousigma': np.maximum(1e-8, npr.normal(loc=0.1, scale=0.05)),
            'lrelu': 10.**npr.uniform(-4, -1),
            'naf_bn': bool(npr.binomial(1, 0.5)),
            'icnn_bn': bool(npr.binomial(1, 0.25)),
        }
        if hp_alg['l2norm'] < 1e-8: hp_alg['l2norm'] = 0.
        if hp_alg['pl2norm'] < 1e-8: hp_alg['pl2norm'] = 0.
        if hp_alg['lrelu'] < 1e-3: hp_alg['lrelu'] = 0.

        for alg in algs:
            algDir = os.path.join(allDir, alg)
            runExp(args, alg, algDir, i, hp_alg)

def runExp(args, alg, algDir, expNum, hp_alg):
    hp = hyperparams[args.task]
    expDir = os.path.join(algDir, str(expNum).zfill(3))

    if os.path.exists(expDir):
        print("==============")
        print("Skipping {}.{}, already exists.".format(alg, expNum))
        print("==============")
        return

    nTestEpisode = 1
    monitor = -1
    seed = 0
    cmd = [pythonCmd, mainSrc, '--model', alg, '--env', args.task+'-v1',
           '--outdir', expDir,
           '--total', str(hp['total']), '--train', str(hp['testInterval']),
           '--test', str(nTestEpisode), '--monitor', str(monitor),
           '--tfseed', str(seed), '--gymseed', str(seed), '--npseed', str(seed)]

    if 'ymin' in hp:
        cmd += ['--ymin', str(hp['ymin'])]
    if 'ymax' in hp:
        cmd += ['--ymax', str(hp['ymax'])]

    for opt, val in hp_alg.items():
        cmd += ['--'+opt, str(val)]

    p = Popen(cmd)
    p.communicate()

hyperparams = {
    'HalfCheetah': {
        'ymin': -500, 'ymax': 5500, 'total': 100000, 'testInterval': 1000,},
    'Hopper': {'ymin': -500, 'ymax': 2500, 'total': 100000, 'testInterval': 1000,},
    'InvertedPendulum': {
        'ymin': 0, 'ymax': 1100, 'total': 10000, 'testInterval': 100,},
    'InvertedDoublePendulum': {
        'ymin': 0, 'ymax': 10000, 'total': 10000, 'testInterval': 100,},
    'Reacher': {'ymin': -15, 'ymax': -5, 'total': 10000, 'testInterval': 100,},
    'Swimmer': {'ymin': -50, 'ymax': 450, 'total': 100000, 'testInterval': 1000,},
    'Walker2d': {'ymin': -50, 'ymax': 450, 'total': 200000, 'testInterval': 2000,},
    'Ant': {'total': 100000, 'testInterval': 1000,},
    'Humanoid': {'total': 100000, 'testInterval': 1000,},
    'HumanoidStandup': {'total': 100000, 'testInterval': 1000,},
}

if __name__=='__main__':
    main()
