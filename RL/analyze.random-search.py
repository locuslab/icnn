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

pythonCmd = 'python3'
rlDir = os.path.dirname(os.path.realpath(__file__))
plotSrc = os.path.join(rlDir, 'src', 'plot.py')
mainSrc = os.path.join(rlDir, 'src', 'main.py')

all_algs = ['DDPG', 'NAF', 'ICNN']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expDir', type=str, default='output.random-search')
    args = parser.parse_args()

    bestParams, bestVals = {}, {}
    for task in os.listdir(args.expDir):
        taskDir = os.path.join(args.expDir, task)
        if os.path.isdir(taskDir):
            bestParams[task], bestVals[task] = analyzeTask(taskDir)

    tableP = os.path.join(args.expDir, 'table.tex')
    with open(tableP, 'w') as f:
        f.write('{:>25s} & DDPG & (episodes) & NAF & (episodes) & ICNN & (episodes) \\\\ \\hline\n'.format('Task'))
        for task, algs in sorted(bestVals.items()):
            vals = [algs[alg][0] for alg in all_algs]
            episodes = [algs[alg][1] for alg in all_algs]
            vals_eps = list(zip(vals, episodes, all_algs))
            s = sorted(vals_eps, key=operator.itemgetter(1))
            s = sorted(s, key=operator.itemgetter(0), reverse=True)
            bestAlg = s[0][2]

            def getStr(alg):
                s = '{:.2f} ({:d})'.format(algs[alg][0], int(algs[alg][1]))
                if alg == bestAlg:
                    s = '\\textbf{' + s + '}'
                return s

            f.write('{:>25s} & {} & {} & {} \\\\\n'.format(
                task, getStr('DDPG'), getStr('NAF'), getStr('ICNN')))
    print('Created {}'.format(tableP))

    tableP = os.path.join(args.expDir, 'table.org')
    with open(tableP, 'w') as f:
        f.write('| Task | DDPG | NAF | ICNN |\n')
        f.write('|------+------+-----+------|\n')
        for task, algs in sorted(bestVals.items()):
            vals = [algs[alg][0] for alg in all_algs]
            episodes = [algs[alg][1] for alg in all_algs]
            vals_eps = list(zip(vals, episodes, all_algs))
            s = sorted(vals_eps, key=operator.itemgetter(1))
            s = sorted(s, key=operator.itemgetter(0), reverse=True)
            bestAlg = s[0][2]

            def getStr(alg):
                s = '{:.2f} ({:d})'.format(algs[alg][0], int(algs[alg][1]))
                if alg == bestAlg:
                    s = '*{}*'.format(s)
                return s

            f.write('| {:s} | {} | {} | {} |\n'.format(
                task, getStr('DDPG'), getStr('NAF'), getStr('ICNN')))
    print('Created {}'.format(tableP))

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

    bestParamsP = os.path.join(args.expDir, 'bestParams.json')
    with open(bestParamsP, 'w') as f:
        json.dump(bestParams, f, indent=2, sort_keys=True)
    print('Created {}'.format(bestParamsP))

def analyzeTask(taskDir):
    bestParams = {}
    bestVals = {}
    print('=== {}'.format(taskDir))
    with open(os.path.join(taskDir, 'analysis.txt'), 'w') as f:
        for alg in all_algs:
            algDir = os.path.join(taskDir, alg)
            if os.path.exists(algDir):
                print('  + {}'.format(alg))
                f.write("\n=== {} ===\n".format(alg))
                bestVal, bestTime, bestExp = [None]*3
                for exp in sorted(os.listdir(algDir)):
                    expDir = os.path.join(algDir, exp)
                    testRew = np.loadtxt(os.path.join(expDir, 'test.log'))
                    vals = testRew[:,1]
                    maxVal, maxValI = vals.max(), vals.argmax()
                    timestep = testRew[maxValI,0]
                    if bestVal is None or maxVal > bestVal or \
                       (maxVal == bestVal and timestep < bestTime):
                        bestVal, bestTime, bestExp = maxVal, timestep, exp
                    f.write('  + Experiment {}: Max test reward of {} at timestep {}\n'.format(exp, maxVal, timestep))

                f.write('\n--- Best reward of {} obtained at timestep {} of experiment {}\n'.format(bestVal, bestTime, bestExp))
                flagsP = os.path.join(algDir, bestExp, 'flags.json')
                with open(flagsP, 'r') as flagsF:
                    f.write(flagsF.read()+'\n')
                    flagsF.seek(0)
                    flags = json.load(flagsF)

                bestParams[alg] = flags
                bestVals[alg] = (bestVal, bestTime, bestExp)

    return bestParams, bestVals

if __name__=='__main__':
    main()
