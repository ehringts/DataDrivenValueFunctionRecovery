import numpy as np
from   functions import  model,data,kernel, surrogate
import io
import contextlib
import warnings

def crossValidationControl(dataAll, surrogate, nFolds, nGreedy):
    nData      = dataAll.state.shape[1]
    indices    = np.random.permutation(np.arange(nData))
    resMaxList = np.zeros(nFolds)
    folds      = np.array_split(indices, nFolds)

    for i, testIndices in enumerate(folds):
        print("Fold " + str(i+1) + "/" + str(nFolds))

        trainIndices = np.setdiff1d(indices, testIndices)

        # Training fold
        newData         = data.TrainData()
        newData.state   = dataAll.state[:, trainIndices]
        newData.costate = dataAll.costate[:, trainIndices]
        newData.vf      = np.atleast_2d(dataAll.vf[0, trainIndices])

        # Test fold
        newDataTest         = data.TrainData()
        newDataTest.state   = dataAll.state[:, testIndices]
        newDataTest.costate = dataAll.costate[:, testIndices]
        newDataTest.vf      = np.atleast_2d(dataAll.vf[0, testIndices])

        # Train control surrogate
        surrogate.doFGreedy(newData, nGreedy, 10**(-16))

        # Build test control-functional data
        y, funcYList, rhs = surrogate.makeControlData(newDataTest)

        # Predict control functionals
        preOut = surrogate.evalGen(y, funcYList)



        res = np.abs(preOut - rhs) 

        resMaxList[i] = np.max(res)

    return np.sum(resMaxList) / nFolds

def crossValidationControlList(dataAll, surrogate, nFolds, nGreedy, gammaList):
    errorList = np.full(len(gammaList), 1e5, dtype=float)  # Default "schlecht"
    for i, gamma in enumerate(gammaList):
        print(f"Gamma {i+1}/{len(gammaList)}: {gamma}")

        surrogate.kernel.gamma = gamma

        
        sink = io.StringIO()
        try:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # optional: Warnings auch unterdrücken
                    errorList[i] = crossValidationControl(dataAll, surrogate, nFolds, nGreedy)
        except Exception as e:
             continue
    errorList[np.isnan(errorList)] = 10**5
  
    return gammaList[np.argmin(errorList)],np.min(errorList),errorList

def crossValidationClassic(dataAll, surrogate, nFolds, nGreedy):
    nData      = dataAll.state.shape[1]
    foldSize   = nData//nFolds
    indices    = np.random.permutation(np.arange(nData))
    resMaxList = np.zeros(nFolds)
    folds      = np.array_split(indices, nFolds)
    
    for i, testIndices in enumerate(folds):
        print("Fold " + str(i+1) + "/" + str(nFolds))
        trainIndices    = np.setdiff1d(indices,testIndices)    
        newData         = data.TrainData()
        newData.state   = dataAll.state[:,trainIndices]
        newData.costate = dataAll.costate[:,trainIndices]  
        newData.vf      = np.atleast_2d(dataAll.vf[0,trainIndices])
        surrogate.doFGreedy(newData,nGreedy,10**(-16))

        N, M          = dataAll.state[:,testIndices].shape
        y             = np.tile(dataAll.state[:,testIndices], (1, N + 1))          # same as repeatedly doing np.c_[Y, S] N times
        funcYList     = np.repeat(np.arange(N + 1), M).astype(int)  # [0..0, 1..1, ..., N..N], each repeated M times       
        rhs           = np.r_[dataAll.vf[0,testIndices],dataAll.costate[:,testIndices].flatten()] # right hand side of the PDE at the data points
        preOut        = surrogate.evalGen(y,funcYList)
        res           = (np.abs(preOut-rhs))
        resMaxList[i] = np.max(res[:M])
    return np.sum(resMaxList)/nFolds   

def crossValidationListClassic(dataAll, surrogate, nFolds, nGreedy, gammaList):
    errorList = np.full(len(gammaList), 1e5, dtype=float)  # Default "schlecht"
    for i, gamma in enumerate(gammaList):
        print(f"Gamma {i+1}/{len(gammaList)}: {gamma}")

        surrogate.kernel.gamma = gamma

        
        sink = io.StringIO()
        try:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # optional: Warnings auch unterdrücken
                    errorList[i] = crossValidationClassic(dataAll, surrogate, nFolds, nGreedy)
        except Exception as e:
             continue
    errorList[np.isnan(errorList)] = 10**5
   
    return gammaList[np.argmin(errorList)],np.min(errorList),errorList

