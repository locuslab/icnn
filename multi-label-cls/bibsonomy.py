import numpy as np
import os
import pickle as pkl

def _loadTxt(fName):
    return np.loadtxt(fName, delimiter=',', skiprows=1, dtype=np.float32)

def cache(dirPath):
    cacheF = os.path.join(dirPath, 'cache.pkl')
    if os.path.isfile(cacheF):
        with open(cacheF, 'rb') as f:
            return pkl.load(f)
    return None

def loadBibtex(dirPath):
    d = cache(dirPath)
    if d is None:
        train = _loadTxt(os.path.join(dirPath, "bibtex-train.csv"))
        trainX, trainY = np.split(train, [1836], axis=1)
        test = _loadTxt(os.path.join(dirPath, "bibtex-test.csv"))
        testX, testY = np.split(test, [1836], axis=1)
        d = {'trainX': trainX, 'trainY': trainY,
            'testX': testX, 'testY': testY}
        with open(cacheF, 'wb') as f:
            pkl.dump(d, f)

    return d

def loadDelicious(dirPath):
    d = cache(dirPath)
    if d is None:
        train = _loadTxt(os.path.join(dirPath, "delicious-train.csv"))
        trainX, trainY = np.split(train, [500], axis=1)
        test = _loadTxt(os.path.join(dirPath, "delicious-test.csv"))
        testX, testY = np.split(test, [500], axis=1)
        d = {'trainX': trainX, 'trainY': trainY,
            'testX': testX, 'testY': testY}
        with open(cacheF, 'wb') as f:
            pkl.dump(d, f)

    return d
