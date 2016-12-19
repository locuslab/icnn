#!/usr/bin/env python3

import argparse
import os
import sys
import shutil
from subprocess import Popen, PIPE

pythonCmd = 'python3'
rlDir = os.path.dirname(os.path.realpath(__file__))
# rlDir = os.path.join(fileDir, '..')
plotSrc = os.path.join(rlDir, 'plot-all.py')
mainSrc = os.path.join(rlDir, 'src', 'main.py')

def main():
    all_algs = ['DDPG', 'NAF', 'ICNN']
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str,
                        choices=['HalfCheetah', 'Hopper', 'InvertedDoublePendulum',
                                 'InvertedPendulum', 'Reacher', 'Swimmer'],
                        help='(Every task is currently v1.)')
    parser.add_argument('--alg', type=str, choices=all_algs)
    parser.add_argument('--nTrials', type=int, default=10)
    parser.add_argument('--save', type=str)
    parser.add_argument('--overwrite', action='store_true')

    args = parser.parse_args()

    allDir = args.save or os.path.join('output', args.task)
    if os.path.exists(allDir):
        if args.overwrite:
            shutil.rmtree(allDir)

    algs = [args.alg] if args.alg is not None else all_algs
    for alg in algs:
        runAlg(args, alg, allDir)

def runAlg(args, alg, allDir):
    algDir = os.path.join(allDir, alg)

    for i in range(args.nTrials):
        runExp(args, alg, algDir, i)
        plot(args, allDir)

def plot(args, allDir):
    pltCmd = [pythonCmd, plotSrc]
    hp = hyperparams[args.task]
    if 'ymin' in hp:
        pltCmd += ['--ymin', str(hp['ymin'])]
    if 'ymax' in hp:
        pltCmd += ['--ymax', str(hp['ymax'])]
    pltCmd.append(allDir)
    os.system(' '.join(pltCmd))

def runExp(args, alg, algDir, seed):
    hp = hyperparams[args.task]
    hp_alg = hp[alg]
    expDir = os.path.join(algDir, str(seed))

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
        'ymin': -500, 'ymax': 5500, 'total': 100000, 'testInterval': 1000,
        'DDPG': {'reward_k': 1.0, 'l2norm': 0.01},
        'NAF': {'reward_k': 0.1, 'initstd': 0.007, 'naf_bn': True},
        'ICNN': {'reward_k': 0.4, 'l2norm': 0.}
    },
    'Hopper': {
        'ymin': -500, 'ymax': 2500, 'total': 100000, 'testInterval': 1000,
        'DDPG': {'reward_k': 1.0, 'l2norm': 0.001},
        'NAF': {'reward_k': 0.3, 'l2norm': 0.001},
        'ICNN': {'reward_k': 0.01, 'l2norm': 0.}
    },
    'InvertedPendulum': {
        'ymin': 0, 'ymax': 1100, 'total': 10000, 'testInterval': 100,
        'DDPG': {'reward_k': 0.3},
        'NAF': {'reward_k': 0.2},
        'ICNN': {'reward_k': 0.3, 'l2norm': 0.}
    },
    'InvertedDoublePendulum': {
        'ymin': 0, 'ymax': 10000, 'total': 10000, 'testInterval': 100,
        'DDPG': {'reward_k': 0.03},
        'NAF': {'reward_k': 0.02},
        'ICNN': {'reward_k': 0.01, 'l2norm': 0.}
    },
    'Reacher': {
        'ymin': -15, 'ymax': -5, 'total': 10000, 'testInterval': 100,
        'DDPG': {'reward_k': 1.,},
        'NAF': {'reward_k': 1.,},
        'ICNN': {'reward_k': 1., 'l2norm': 0.}
    },
    'Swimmer': {
        'ymin': -50, 'ymax': 450, 'total': 100000, 'testInterval': 1000,
        'DDPG': {'reward_k': 1.,},
        'NAF': {'reward_k': 1.,},
        'ICNN': {'reward_k': 1., 'l2norm': 0.}
    },
    'Walker2d': {
        'ymin': -50, 'ymax': 450, 'total': 200000, 'testInterval': 2000,
        'DDPG': {'reward_k': 0.1,},
        'NAF': {'reward_k': 0.1,},
        'ICNN': {'reward_k': 0.04, 'l2norm': 0.}
    },
    'Ant': {
        # TODO
        'total': 100000, 'testInterval': 1000,
        'DDPG': {'reward_k': 0.2,},
        'NAF': {'reward_k': 0.3,},
        'ICNN': {'reward_k': 1., 'l2norm': 0.}
    },
    'Humanoid': {
        # TODO
        'total': 100000, 'testInterval': 1000,
        'DDPG': {'reward_k': 0.03, 'l2norm': 0.01},
        'NAF': {'reward_k': 0.01,},
        'ICNN': {'reward_k': 1., 'l2norm': 0.}
    },
    'HumanoidStandup': {
        # TODO
        'total': 100000, 'testInterval': 1000,
        'DDPG': {'reward_k': 1.,},
        'NAF': {'reward_k': 1.,},
        'ICNN': {'reward_k': 1., 'l2norm': 0.}
    },
}

if __name__=='__main__':
    main()
