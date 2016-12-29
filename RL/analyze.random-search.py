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
plotSrc = os.path.join(rlDir, 'plot-all.py')
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
        f.write('{:>25s} & DDPG & & NAF & & ICNN & \\\\ \\hline\n'.format('Task'))
        for task, algs in sorted(bestVals.items()):
            bestAlg = sorted(algs.items(), key=operator.itemgetter(1),
                             reverse=True)[0][0]

            def getStr(alg):
                s = '{:.2f} ({:.2f})'.format(algs[alg][0], int(algs[alg][1]))
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
            bestAlg = sorted(algs.items(), key=operator.itemgetter(1),
                             reverse=True)[0][0]

            def getStr(alg):
                s = '{:.2f} ({:.2f})'.format(algs[alg][0], int(algs[alg][1]))
                if alg == bestAlg:
                    s = '*{}*'.format(s)
                return s

            f.write('| {:s} | {} | {} | {} |\n'.format(
                task, getStr('DDPG'), getStr('NAF'), getStr('ICNN')))
            f.flush()
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
                f.write('\n=== {}\n\n'.format(alg))
                exps = {}
                for exp in sorted(os.listdir(algDir)):
                    expDir = os.path.join(algDir, exp)
                    testData = np.loadtxt(os.path.join(expDir, 'test.log'))
                    testRew = testData[:,1]
                    if np.any(np.isnan(testRew)):
                        continue

                    N = 10
                    testRew_ = np.array([sum(testRew[i-N:i])/N for
                                         i in range(N, len(testRew))])
                    exps[exp] = [testRew_[-1], testRew_.sum()]

                    f.write(('  + Experiment {}: Final rolling reward of {} '+
                             'with a cumulative reward of {}\n').format(
                                 *([exp] + exps[exp])))

                s = sorted(exps.items(), key=operator.itemgetter(1), reverse=True)
                best = s[0]
                bestExp = best[0]

                f.write('\n--- Best of {} obtained in experiment {}\n'.format(
                    best[1], bestExp))
                flagsP = os.path.join(algDir, bestExp, 'flags.json')
                with open(flagsP, 'r') as flagsF:
                    f.write(flagsF.read()+'\n')
                    flagsF.seek(0)
                    flags = json.load(flagsF)

                bestParams[alg] = flags
                bestVals[alg] = best[1]

    return bestParams, bestVals

if __name__=='__main__':
    main()
