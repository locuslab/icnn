import numpy as np
import sklearn
from sklearn.metrics import f1_score

def macroF1(trueY, predY):
    # trueY and predY should be (nExamples, nLabels)
    predY_bin = (predY >= 0.5).astype(np.int)
    trueY_bin = trueY.astype(np.int)
    # The transpose is here because sklearn's f1_score expects multi-label
    # data to be formatted as (nLabels, nExamples).
    return f1_score(trueY_bin.T, predY_bin.T, average='macro', pos_label=None)
