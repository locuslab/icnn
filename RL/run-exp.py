#!/usr/bin/env python3

import argparse
import json
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
    parser.add_argument('--bestParams', type=str,
                        default='output.random-search/bestParams.json')
    parser.add_argument('--overwrite', action='store_true')

    args = parser.parse_args()

    with open(args.bestParams, 'r') as f:
        bestParams = json.load(f)

    allDir = args.save or os.path.join('output', args.task)
    if os.path.exists(allDir):
        if args.overwrite:
            shutil.rmtree(allDir)

    algs = [args.alg] if args.alg is not None else all_algs
    for alg in algs:
        runAlg(args, alg, allDir, bestParams)

def runAlg(args, alg, allDir, bestParams):
    algDir = os.path.join(allDir, alg)

    for i in range(args.nTrials):
        runExp(args, alg, algDir, i, bestParams)
        plot(args, allDir)

def plot(args, allDir):
    pltCmd = [pythonCmd, plotSrc, allDir]
    os.system(' '.join(pltCmd))

def runExp(args, alg, algDir, seed, bestParams):
    hp = bestParams[args.task][alg]
    expDir = os.path.join(algDir, str(seed))

    if os.path.exists(expDir):
        print("==============")
        print("Skipping {}.{}, already exists.".format(alg, seed))
        print("==============")
        return

    monitor = -1
    cmd = [pythonCmd, mainSrc, '--model', alg, '--env', args.task+'-v1',
           '--outdir', expDir, '--monitor', str(monitor),
           '--tfseed', str(seed), '--gymseed', str(seed), '--npseed', str(seed)]

    for opt, val in hp.items():
        cmd += ['--'+opt, str(val)]

    p = Popen(cmd)
    p.communicate()

if __name__=='__main__':
    main()
