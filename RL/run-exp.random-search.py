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
plotSrc = os.path.join(rlDir, 'src', 'plot.py')
mainSrc = os.path.join(rlDir, 'src', 'main.py')

all_algs = ['DDPG', 'NAF', 'ICNN']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str,
                        choices=['HalfCheetah', 'Hopper', 'InvertedDoublePendulum',
                                 'InvertedPendulum', 'Reacher', 'Swimmer'],
                        help='(Every task is currently v1.)')
    parser.add_argument('--alg', type=str, choices=all_algs)
    parser.add_argument('--nSamples', type=int, default=10)
    parser.add_argument('--save', type=str)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--analyze', action='store_true')

    args = parser.parse_args()

    allDir = args.save or os.path.join('output.random-search', args.task)
    if os.path.exists(allDir):
        if args.overwrite:
            shutil.rmtree(allDir)

    if args.analyze:
        analyze(args, allDir)
    else:
        algs = [args.alg] if args.alg is not None else all_algs
        for alg in algs:
            runAlg(args, alg, allDir)

def runAlg(args, alg, allDir):
    algDir = os.path.join(allDir, alg)

    np.random.seed(0)
    for i in range(args.nSamples):
        seed = 0
        reward_k = 10.**npr.randint(-3, 0+1)
        l2norm = 10.**npr.randint(-10, -3+1)
        if l2norm < 1e-8:
            l2norm = 0.
        runExp(args, alg, algDir, seed, reward_k, l2norm)
        analyze(args, allDir)

def analyze(args, allDir):
    with open(os.path.join(allDir, 'analysis.txt'), 'w') as f:
        f.write("Format: [experiment, max test loss]\n")
        for alg in all_algs:
            algDir = os.path.join(allDir, alg)
            if os.path.exists(algDir):
                f.write("\n=== {} ===\n".format(alg))
                for exp in os.listdir(algDir):
                    expDir = os.path.join(algDir, exp)
                    testLoss = np.loadtxt(os.path.join(expDir, 'test.log'))
                    f.write('  + {}: {}\n'.format(exp, testLoss[:,1].max()))

def runExp(args, alg, algDir, seed, reward_k=1., l2norm=0.):
    hp = hyperparams[args.task]
    hp_alg = {'reward_k': reward_k, 'l2norm': l2norm}
    expDir = os.path.join(algDir, "seed={},reward_k={:.0e},l2norm={:.0e}".format(
        seed, reward_k, l2norm))

    if os.path.exists(expDir):
        print("==============")
        print("Skipping {}.{}, already exists.".format(alg, seed))
        print("==============")
        return

    nTestEpisode = 1
    monitor = -1
    cmd = [pythonCmd, mainSrc, '--model', alg, '--env', args.task+'-v1',
           '--outdir', expDir,
           '--total', str(hp['total']), '--train', str(hp['testInterval']),
           '--test', str(nTestEpisode), '--monitor', str(monitor),
           '--tfseed', str(seed), '--gymseed', str(seed), '--npseed', str(seed)]

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
    'Walker2d': {'ymin': -50, 'ymax': 450, 'total': 10000, 'testInterval': 100,},
    'Ant': {'total': 100000, 'testInterval': 1000,},
    'Humanoid': {'total': 100000, 'testInterval': 1000,},
    'HumanoidStandup': {'total': 100000, 'testInterval': 1000,},
}

if __name__=='__main__':
    main()
