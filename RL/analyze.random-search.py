#!/usr/bin/env python3

import argparse
import os
import sys
import shutil
from subprocess import Popen, PIPE
import json
import operator

import numpy as np
import numpy.random as npr

import pandas as pd

pythonCmd = 'python3'
rlDir = os.path.dirname(os.path.realpath(__file__))
plotSrc = os.path.join(rlDir, 'plot-all.py')
mainSrc = os.path.join(rlDir, 'src', 'main.py')

all_algs = ['DDPG', 'NAF', 'ICNN']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expDir', type=str, default='output.random-search')
    args = parser.parse_args()

    task_dfs = {}
    for task in os.listdir(args.expDir):
        taskDir = os.path.join(args.expDir, task)
        if os.path.isdir(taskDir):
            alg_dfs = analyzeTask(taskDir)
            task_dfs[task] = alg_dfs

    makeTable(args, 'cumulativeTestRew', task_dfs)
    makeTable(args, 'maxTestRew', task_dfs)
    makeTable(args, 'finalTestRew', task_dfs)

    # bestParamsP = os.path.join(args.expDir, 'bestParams.json')
    # with open(bestParamsP, 'w') as f:
    #     json.dump(bestParams, f, indent=2, sort_keys=True)
    # print('Created {}'.format(bestParamsP))

def makeTable(args, tag, task_dfs):
    orgTableP = os.path.join(args.expDir, '{}.table.org'.format(tag))
    with open(orgTableP, 'w') as f:
        f.write('| Task | DDPG | NAF | ICNN |\n')
        f.write('|------+------+-----+------|\n')
        for task, alg_dfs in sorted(task_dfs.items()):
            stats  = []
            for alg in all_algs:
                exps = alg_dfs[alg]
                stats.append(exps[tag].max())
            bestStat = max(stats)

            def getStr(val):
                s = '{:.2f}'.format(val)
                if (bestStat - val)/np.abs(bestStat) <= 0.01:
                    s = '*{}*'.format(s)
                return s

            f.write('| {:s} | {} | {} | {} |\n'.format(
                *([task]+list(map(getStr, stats)))))
            f.flush()
    print('Created {}'.format(orgTableP))

    texTableP = os.path.join(args.expDir, '{}.table.tex'.format(tag))
    os.system('pandoc {} --to latex --output {}'.format(orgTableP, texTableP))


def analyzeTask(taskDir):
    alg_dfs = {}
    for alg in all_algs:
        algDir = os.path.join(taskDir, alg)
        if os.path.exists(algDir):
            exps = []
            for exp in sorted(os.listdir(algDir)):
                expDir = os.path.join(algDir, exp)
                testData = np.loadtxt(os.path.join(expDir, 'test.log'))
                testRew = testData[:,1]

                N = 10

                if np.any(np.isnan(testRew)) or testRew.size <= N:
                    continue

                testRew = rolling(N, testRew)
                d = {
                    'exp': exp,
                    'maxTestRew': testRew.max(),
                    'finalTestRew': testRew[-1],
                    'cumulativeTestRew': testRew.sum()
                }
                exps.append(d)

            df = pd.DataFrame(exps)
            df = df.set_index('exp')
            alg_dfs[alg] = df

    return alg_dfs

def rolling(N, rew):
    K = np.full(N, 1./N)
    return np.convolve(rew, K, 'valid')


def getParams(p):
    # params = getParams(os.path.join(algDir, bestExp, 'flags.json'))
    with open(flagsP, 'r') as flagsF:
        f.write(flagsF.read()+'\n')
        flagsF.seek(0)
        flags = json.load(flagsF)

    assert(False)
    for task, algs in bestParams.items():
        for alg, params in algs.items():
            del params['copy']
            del params['env']
            del params['force']
            del params['gdb']
            del params['gymseed']
            del params['model']
            del params['monitor']
            del params['npseed']
            del params['outdir']
            del params['summary']
            del params['tfseed']
            del params['thread']

if __name__=='__main__':
    main()
