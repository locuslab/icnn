#!/usr/bin/env python3

import numpy as np
import numpy.random as npr

import argparse
import csv
import os
import sys
import time
import pickle as pkl
import json
import shutil

import setproctitle

from datetime import datetime

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import olivetti

from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nEpoch', type=float, default=10000)
    parser.add_argument('--trainBatchSz', type=int, default=70)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save', type=str, default='work/baseline')

    args = parser.parse_args()
    setproctitle.setproctitle('bamos.icnn.comp.baseline')

    save = os.path.expanduser(args.save)
    if os.path.isdir(save):
        shutil.rmtree(save)
    os.makedirs(save)

    npr.seed(args.seed)
    torch.manual_seed(args.seed)

    data = olivetti.load("data/olivetti")
    data = {
        k: torch.tensor(v).transpose(1,3).float().cuda()
        for (k,v) in data.items()
    }

    train_data = TensorDataset(data['trainX'], data['trainY'])
    train_loader = DataLoader(
        train_data,
        batch_size=args.trainBatchSz,
        shuffle=True
    )

    nTrain = data['trainX'].shape[0]
    nTest = data['testX'].shape[0]

    inputSz = list(data['trainX'][0].shape)
    outputSz = list(data['trainY'][1].shape)

    print("\n\n" + "="*40)
    print("+ nTrain: {}, nTest: {}".format(nTrain, nTest))
    print("+ inputSz: {}, outputSz: {}".format(inputSz, outputSz))
    print("="*40 + "\n\n")

    model = FCN().cuda()
    # model = DilatedModel().cuda()
    opt = optim.Adam(model.parameters(), lr=1e-4)

    trainFields = ['iter', 'loss']
    trainF = open(os.path.join(save, 'train.csv'), 'w')
    trainW = csv.writer(trainF)
    trainW.writerow(trainFields)

    testFields = ['iter', 'loss']
    testF = open(os.path.join(save, 'test.csv'), 'w')
    testW = csv.writer(testF)
    testW.writerow(testFields)

    best_test_loss = None
    for epoch in range(args.nEpoch):
        for x, y in train_loader:
            yhat = model(x)
            loss = (255*(yhat - y)).pow(2).mean()
            trainW.writerow((-1, loss.item()))
            trainF.flush()

            opt.zero_grad()
            loss.backward()
            opt.step()

        yhat = model(data['testX'])
        test_loss = (255*(yhat - data['testY'])).pow(2).mean()
        testW.writerow((epoch, test_loss.item()))
        testF.flush()
        if best_test_loss is None or test_loss < best_test_loss:
            best_test_loss = test_loss.item()
            print('+ Best test loss: ', best_test_loss)

class FCN(nn.Module):
    def __init__(self, k=32):
        super(FCN, self).__init__()

        self.conv1 = nn.Conv2d(1, k, 3, stride=2, dilation=2, padding=2)
        self.conv2 = nn.Conv2d(k, k, 3, stride=2, dilation=2, padding=2)
        self.conv3 = nn.Conv2d(k, k, 3, stride=2, dilation=2, padding=2)

        self.up1 = nn.ConvTranspose2d(
            k, k, 3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(
            k, k, 3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose2d(
            k, k, 3, stride=2, padding=1, output_padding=1)
        self.up4 = nn.ConvTranspose2d(k, 1, 1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.up1(h))
        h = F.relu(self.up2(h))
        h = F.relu(self.up3(h))
        h = self.up4(h)
        assert h.shape == x.shape
        return h

class DilatedModel(nn.Module):
    def __init__(self, k=16):
        super(DilatedModel, self).__init__()

        self.conv1 = nn.Conv2d(1, k, 3, stride=1, dilation=1, padding=1)
        self.conv2 = nn.Conv2d(k, k, 3, stride=1, dilation=1, padding=1)
        self.conv3 = nn.Conv2d(k, k, 3, stride=1, dilation=2, padding=2)
        self.conv4 = nn.Conv2d(k, k, 3, stride=1, dilation=4, padding=4)
        self.conv5 = nn.Conv2d(k, k, 3, stride=1, dilation=8, padding=8)
        self.conv6 = nn.Conv2d(k, k, 3, stride=1, dilation=16, padding=16)
        self.conv7 = nn.Conv2d(k, k, 3, stride=1, dilation=1, padding=1)
        self.conv8 = nn.Conv2d(k, 1, 1, stride=1, dilation=1, padding=0)

    def forward(self, x):
        h = x
        h = F.relu(self.conv1(h))
        assert h.shape[2:] == x.shape[2:]
        h = F.relu(self.conv2(h))
        assert h.shape[2:] == x.shape[2:]
        h = F.relu(self.conv3(h))
        assert h.shape[2:] == x.shape[2:]
        h = F.relu(self.conv3(h))
        assert h.shape[2:] == x.shape[2:]
        h = F.relu(self.conv4(h))
        assert h.shape[2:] == x.shape[2:]
        h = F.relu(self.conv5(h))
        assert h.shape[2:] == x.shape[2:]
        h = F.relu(self.conv6(h))
        assert h.shape[2:] == x.shape[2:]
        h = F.relu(self.conv7(h))
        assert h.shape[2:] == x.shape[2:]
        h = F.relu(self.conv8(h))
        assert h.shape == x.shape
        return h

if __name__=='__main__':
    main()
