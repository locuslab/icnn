# local image = require 'image'

# local H = require 'icnn.helper'

# function f(inputFile, hidden, nTraining)
#    local data = {}
#    data.all = H.loadtxt(inputFile)
#       :t():clone()
#       :view(400, 1, 64, 64)
#       :transpose(3, 4):clone()
#       :div(255)

#    local topI = {{}, {}, {1, 32}, {1, 64}}
#    local bottomI = {{}, {}, {33, 64}, {1, 64}}
#    local leftI = {{}, {}, {1, 64}, {1, 32}}
#    local rightI = {{}, {}, {1, 64}, {33, 64}}
#    local trainI = {{1, nTraining}}
#    local testI = {{nTraining+1, 400}}

#    if hidden == 'left' then
#       data.inputSz = torch.IntTensor{1, 64, 32}
#       data.outputSz = data.inputSz

#       data.trainX = cast(data.all[trainI][rightI])
#       data.trainY = cast(data.all[trainI][leftI])
#       data.testX = cast(data.all[testI][rightI])
#       data.testY = cast(data.all[testI][leftI])

#       data.trainXflip = cast(image.hflip(data.all[trainI][rightI]))
#       data.testXflip = cast(image.hflip(data.all[testI][rightI]))
#    elseif hidden == 'bottom' then
#       data.inputSz = torch.IntTensor{1, 32, 64}
#       data.outputSz = data.inputSz

#       data.trainX = cast(data.all[trainI][topI])
#       data.trainY = cast(data.all[trainI][bottomI])
#       data.testX = cast(data.all[testI][topI])
#       data.testY = cast(data.all[testI][bottomI])
#    else
#       assert(false)
#    end

#    data.testYflat = data.testY:clone():view(data.testY:size(1), -1)

#    data.avgY = data.trainY:mean(1):view(
#       data.trainY:size(2), data.trainY:size(3), data.trainY:size(4))
#    return data
# end

# return f

import numpy as np
import pickle as pkl
import os

def load(prefix):
    cacheF = prefix + '.pkl'
    if os.path.isfile(cacheF):
        print('olivetti: loading from cache')
        with open(cacheF, 'rb') as f:
            d = pkl.load(f)
    else:
        print('olivetti: Creating from txt')
        data = np.loadtxt(prefix + '.raw')
        data = data.T.reshape((400,64,64,1)).transpose((0,2,1,3))/255.
        d = {'trainX': data[0:350,:,32:64,:],
             'trainY': data[0:350,:,0:32,:],
             'testX': data[350:400,:,32:64,:],
             'testY': data[350:400,:,0:32,:]}
        with open(cacheF, 'wb') as f:
            pkl.dump(d, f)

    return d
