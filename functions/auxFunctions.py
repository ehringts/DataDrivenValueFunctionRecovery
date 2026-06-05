"""Helper routines for cross-validation of the classical and direct-control surrogates."""

import numpy as np
from functions import data
import io
import contextlib
import warnings


def crossValidationControl(dataAll, surrogate, nFolds, nGreedy):
    """Run k-fold cross-validation for the direct-control surrogate and average the maximum validation error."""
    nData      = dataAll.state.shape[1]
    indices    = np.random.permutation(np.arange(nData))
    resMaxList = np.zeros(nFolds)
    folds      = np.array_split(indices, nFolds)
    for i, testIndices in enumerate(folds):
        print('Fold ' + str(i + 1) + '/' + str(nFolds))
        trainIndices        = np.setdiff1d(indices, testIndices)
        newData             = data.TrainData()
        newData.state       = dataAll.state[:, trainIndices]
        newData.costate     = dataAll.costate[:, trainIndices]
        newData.vf          = np.atleast_2d(dataAll.vf[0, trainIndices])
        newDataTest         = data.TrainData()
        newDataTest.state   = dataAll.state[:, testIndices]
        newDataTest.costate = dataAll.costate[:, testIndices]
        newDataTest.vf      = np.atleast_2d(dataAll.vf[0, testIndices])
        surrogate.doFGreedy(newData, nGreedy, 10 ** (-16))
        y, funcYList, rhs = surrogate.makeControlData(newDataTest)
        preOut            = surrogate.evalGen(y, funcYList)
        res               = np.abs(preOut - rhs)
        resMaxList[i]     = np.max(res)
    return np.sum(resMaxList) / nFolds


def crossValidationControlList(dataAll, surrogate, nFolds, nGreedy, gammaList):
    """Test a list of gamma values for the direct-control surrogate and return the best value and error list."""
    errorList = np.full(len(gammaList), 100000.0, dtype=float)
    for i, gamma in enumerate(gammaList):
        print(f'Gamma {i + 1}/{len(gammaList)}: {gamma}')
        surrogate.kernel.gamma = gamma
        sink                   = io.StringIO()
        try:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    errorList[i] = crossValidationControl(dataAll, surrogate, nFolds, nGreedy)
        except Exception:
            continue
    errorList[np.isnan(errorList)] = 10 ** 5
    return (gammaList[np.argmin(errorList)], np.min(errorList), errorList)


def crossValidationClassic(dataAll, surrogate, nFolds, nGreedy):
    """Run k-fold cross-validation for the classical surrogate and average the maximum validation error."""
    nData      = dataAll.state.shape[1]
    indices    = np.random.permutation(np.arange(nData))
    resMaxList = np.zeros(nFolds)
    folds      = np.array_split(indices, nFolds)
    for i, testIndices in enumerate(folds):
        print('Fold ' + str(i + 1) + '/' + str(nFolds))
        trainIndices    = np.setdiff1d(indices, testIndices)
        newData         = data.TrainData()
        newData.state   = dataAll.state[:, trainIndices]
        newData.costate = dataAll.costate[:, trainIndices]
        newData.vf      = np.atleast_2d(dataAll.vf[0, trainIndices])
        surrogate.doFGreedy(newData, nGreedy, 10 ** (-16))
        N, M          = dataAll.state[:, testIndices].shape
        y             = np.tile(dataAll.state[:, testIndices], (1, N + 1))
        funcYList     = np.repeat(np.arange(N + 1), M).astype(int)
        rhs           = np.r_[dataAll.vf[0, testIndices], dataAll.costate[:, testIndices].flatten()]
        preOut        = surrogate.evalGen(y, funcYList)
        res           = np.abs(preOut - rhs)
        resMaxList[i] = np.max(res[:M])
    return np.sum(resMaxList) / nFolds


def crossValidationListClassic(dataAll, surrogate, nFolds, nGreedy, gammaList):
    """Test a list of gamma values for the classical surrogate and return the best value and error list."""
    errorList = np.full(len(gammaList), 100000.0, dtype=float)
    for i, gamma in enumerate(gammaList):
        print(f'Gamma {i + 1}/{len(gammaList)}: {gamma}')
        surrogate.kernel.gamma = gamma
        sink                   = io.StringIO()
        try:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    errorList[i] = crossValidationClassic(dataAll, surrogate, nFolds, nGreedy)
        except Exception:
            continue
    errorList[np.isnan(errorList)] = 10 ** 5
    return (gammaList[np.argmin(errorList)], np.min(errorList), errorList)
