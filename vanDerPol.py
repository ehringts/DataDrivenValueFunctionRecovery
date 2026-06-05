"""Experiment script for the Van der Pol oscillator: data generation, cross-validation, training, and plotting."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from functions import model, data, kernel, surrogate, auxFunctions

newModel = model.VanDerPol(1, 0.2)
endT     = 20


def makeAllData():
    """Generate the initial-state grid, compute training, test, and long-horizon data, and save the datasets."""
    x                 = np.linspace(-1, 1, 100)
    X, Y              = np.meshgrid(x, x)
    omega             = np.c_[X.flatten(), Y.flatten()].T
    newDataTrain      = data.TrainData()
    newDataTest       = data.TestData()
    newDataTestLongHo = data.TestData()
    newDataTrain.makeTrainData(newModel, endT, endT / 10, 100, omega, 0.05, 1000, np.array([[-1, 1], [-1, 1]]), saveName='trainDataVanDerPol', BVP=True)
    newDataTest.makeTestData(newModel, endT * 10, endT / 10, 100, omega, 100, 54248, 'testDataVanDerPol', BVP=True)
    newDataTestLongHo.makeTestData(newModel, endT * 10, endT / 10, 100, omega / 2, 10, 94642542, 'testDataVanDerPolLongHorizon', BVP=True)


def doCrossValidationControl():
    """Load the training data, run gamma cross-validation for the direct-control surrogate, and print the best result."""
    newDataTrain      = data.TrainData('trainDataVanDerPol')
    gammaList         = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4]
    nFolds            = 5
    iterMax           = 100
    newSurrogate      = surrogate.SurrogateControl(kernel.LinMatern(1.29), newModel)
    minGamm1, min1, _ = auxFunctions.crossValidationControlList(newDataTrain, newSurrogate, nFolds, iterMax, gammaList)
    newSurrogate      = surrogate.SurrogateControl(kernel.LinMaternProduct(1.29), newModel)
    minGamm2, min2, _ = auxFunctions.crossValidationControlList(newDataTrain, newSurrogate, nFolds, iterMax, gammaList)
    newSurrogate      = surrogate.SurrogateControl(kernel.QuadMatern(1.29), newModel)
    minGamm3, min3, _ = auxFunctions.crossValidationControlList(newDataTrain, newSurrogate, nFolds, iterMax, gammaList)
    newSurrogate      = surrogate.SurrogateControl(kernel.QuadMaternProduct(1.29), newModel)
    minGamm4, min4, _ = auxFunctions.crossValidationControlList(newDataTrain, newSurrogate, nFolds, iterMax, gammaList)
    newSurrogate      = surrogate.SurrogateControl(kernel.CubMatern(1.29), newModel)
    minGamm5, min5, _ = auxFunctions.crossValidationControlList(newDataTrain, newSurrogate, nFolds, iterMax, gammaList)
    newSurrogate      = surrogate.SurrogateControl(kernel.CubMaternProduct(1.29), newModel)
    minGamm6, min6, _ = auxFunctions.crossValidationControlList(newDataTrain, newSurrogate, nFolds, iterMax, gammaList)
    newSurrogate      = surrogate.SurrogateControl(kernel.QuartMatern(1.29), newModel)
    minGamm7, min7, _ = auxFunctions.crossValidationControlList(newDataTrain, newSurrogate, nFolds, iterMax, gammaList)
    newSurrogate      = surrogate.SurrogateControl(kernel.QuartMaternProduct(1.29), newModel)
    minGamm8, min8, _ = auxFunctions.crossValidationControlList(newDataTrain, newSurrogate, nFolds, iterMax, gammaList)
    print('Best gamma for LinMatern: ' + str(minGamm1) + ' with error: ' + str(min1))
    print('Best gamma for LinMaternProduct: ' + str(minGamm2) + ' with error: ' + str(min2))
    print('Best gamma for QuadMatern: ' + str(minGamm3) + ' with error: ' + str(min3))
    print('Best gamma for QuadMaternProduct: ' + str(minGamm4) + ' with error: ' + str(min4))
    print('Best gamma for CubMatern: ' + str(minGamm5) + ' with error: ' + str(min5))
    print('Best gamma for CubMaternProduct: ' + str(minGamm6) + ' with error: ' + str(min6))
    print('Best gamma for QuartMatern: ' + str(minGamm7) + ' with error: ' + str(min7))
    print('Best gamma for QuartMaternProduct: ' + str(minGamm8) + ' with error: ' + str(min8))


def doCrossValidationClassic():
    """Load the training data, run gamma cross-validation for the classical value-function surrogate, and print the best result."""
    newDataTrain      = data.TrainData('trainDataVanDerPol')
    gammaList         = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]
    nFolds            = 5
    iterMax           = 100
    newSurrogate      = surrogate.SurrogateClassic(kernel.LinMatern(1.29))
    minGamm1, min1, _ = auxFunctions.crossValidationListClassic(newDataTrain, newSurrogate, nFolds, iterMax, gammaList)
    newSurrogate      = surrogate.SurrogateClassic(kernel.LinMaternProduct(1.29))
    minGamm2, min2, _ = auxFunctions.crossValidationListClassic(newDataTrain, newSurrogate, nFolds, iterMax, gammaList)
    newSurrogate      = surrogate.SurrogateClassic(kernel.QuadMatern(1.29))
    minGamm3, min3, _ = auxFunctions.crossValidationListClassic(newDataTrain, newSurrogate, nFolds, iterMax, gammaList)
    newSurrogate      = surrogate.SurrogateClassic(kernel.QuadMaternProduct(1.29))
    minGamm4, min4, _ = auxFunctions.crossValidationListClassic(newDataTrain, newSurrogate, nFolds, iterMax, gammaList)
    newSurrogate      = surrogate.SurrogateClassic(kernel.CubMatern(1.29))
    minGamm5, min5, _ = auxFunctions.crossValidationListClassic(newDataTrain, newSurrogate, nFolds, iterMax, gammaList)
    newSurrogate      = surrogate.SurrogateClassic(kernel.CubMaternProduct(1.29))
    minGamm6, min6, _ = auxFunctions.crossValidationListClassic(newDataTrain, newSurrogate, nFolds, iterMax, gammaList)
    newSurrogate      = surrogate.SurrogateClassic(kernel.QuartMatern(1.29))
    minGamm7, min7, _ = auxFunctions.crossValidationListClassic(newDataTrain, newSurrogate, nFolds, iterMax, gammaList)
    newSurrogate      = surrogate.SurrogateClassic(kernel.QuartMaternProduct(1.29))
    minGamm8, min8, _ = auxFunctions.crossValidationListClassic(newDataTrain, newSurrogate, nFolds, iterMax, gammaList)
    print('Best gamma for LinMatern: ' + str(minGamm1) + ' with error: ' + str(min1))
    print('Best gamma for LinMaternProduct: ' + str(minGamm2) + ' with error: ' + str(min2))
    print('Best gamma for QuadMatern: ' + str(minGamm3) + ' with error: ' + str(min3))
    print('Best gamma for QuadMaternProduct: ' + str(minGamm4) + ' with error: ' + str(min4))
    print('Best gamma for CubMatern: ' + str(minGamm5) + ' with error: ' + str(min5))
    print('Best gamma for CubMaternProduct: ' + str(minGamm6) + ' with error: ' + str(min6))
    print('Best gamma for QuartMatern: ' + str(minGamm7) + ' with error: ' + str(min7))
    print('Best gamma for QuartMaternProduct: ' + str(minGamm8) + ' with error: ' + str(min8))


def computeErrorsControl():
    """Train the selected direct-control surrogate, record greedy, test, and long-horizon errors, and save the surrogate."""
    newkernel         = kernel.QuartMaternProduct(0.3)
    newSurrogate      = surrogate.SurrogateControl(newkernel, newModel)
    newDataTrain      = data.TrainData('trainDataVanDerPol')
    newDataTest       = data.TestData('testDataVanDerPol')
    newDataTestLongHo = data.TestData('testDataVanDerPolLongHorizon')
    newSurrogate.doFGreedyWithError(endT * 10, newDataTrain, 100, newDataTest, newDataTestLongHo)
    newSurrogate.saveSr('surrogateVanDerPolControl')


def computeErrorsClassic():
    """Train the selected classical surrogate, record greedy, test, and long-horizon errors, and save the surrogate."""
    newkernel         = kernel.QuartMaternProduct(0.4)
    newSurrogate      = surrogate.SurrogateClassic(newkernel)
    newDataTrain      = data.TrainData('trainDataVanDerPol')
    newDataTest       = data.TestData('testDataVanDerPol')
    newDataTestLongHo = data.TestData('testDataVanDerPolLongHorizon')
    newSurrogate.doFGreedyWithError(newModel, endT * 10, newDataTrain, 100, newDataTest, newDataTestLongHo)
    newSurrogate.saveSr('surrogateVanDerPolClassic')


def plotResult():
    """Load the saved surrogates, plot greedy, test, and performance errors, and save the plot and legend."""
    rc = {'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman', 'CMU Serif', 'DejaVu Serif'], 'axes.titlesize': 18, 'axes.labelsize': 16, 'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12, 'lines.linewidth': 1.6, 'text.latex.preamble': '\\usepackage{amsmath,amssymb}'}
    with plt.rc_context(rc):
        newkernel           = kernel.GaussProduct(1.07)
        newSurrogateControl = surrogate.SurrogateControl(newkernel, newModel)
        newSurrogateControl.loadSr('surrogateVanDerPolControl')
        newSurrogateClassic = surrogate.SurrogateClassic(newkernel)
        newSurrogateClassic.loadSr('surrogateVanDerPolClassic')
        wid     = 1.6
        fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
        colors  = {'Greedy-Error': 'C0', 'Test-Error': 'C1', 'Performance-Error': 'C2'}
        series  = [('Greedy-Error', newSurrogateClassic.trainError, '-', 'Classical', 1, True), ('Test-Error', newSurrogateClassic.testError, '-', 'Classical', 1, True), ('Performance-Error', newSurrogateClassic.testErrorLongHo, '-', 'Classical', 1, False), ('Greedy-Error', newSurrogateControl.trainError, '--', 'Control', 1, True), ('Test-Error', newSurrogateControl.testError, '--', 'Control', 1, True), ('Performance-Error', newSurrogateControl.testErrorLongHo, '--', 'Control', 1, False)]
        xmax    = 1
        for metric, y, ls, method, x_start, drop_first in series:
            y = np.asarray(y)
            if drop_first:
                y = y[1:]
            if len(y) == 0:
                continue
            x    = np.arange(x_start, x_start + len(y))
            xmax = max(xmax, x[-1])
            ax.plot(x, y, lw=wid, ls=ls, color=colors[metric], label=f'{metric} ({method})')
        ax.set_yscale('log')
        ax.set_title('Van der Pol oscillator (\\texttt{VanDerPol})', pad=10)
        ax.set_xlim(1, min(200, xmax))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which='major', alpha=0.6, linewidth=0.8)
        ax.set_axisbelow(True)
        fig.savefig('VanDerPol.pdf', bbox_inches='tight')
        handles, labels = ax.get_legend_handles_labels()
        legend_order    = [0, 3, 1, 4, 2, 5]
        handles         = [handles[i] for i in legend_order]
        labels          = [labels[i] for i in legend_order]
        fig_leg, ax_leg = plt.subplots(figsize=(9, 1.3), constrained_layout=True)
        ax_leg.axis('off')
        ax_leg.legend(handles, labels, loc='center', ncol=3, frameon=False, handlelength=3.0, columnspacing=1.4, handletextpad=0.6, borderaxespad=0.0)
        fig_leg.patch.set_facecolor('white')
        fig_leg.patch.set_edgecolor('black')
        fig_leg.patch.set_linewidth(1.2)
        fig_leg.savefig('legend.pdf', bbox_inches='tight', pad_inches=0.05)
        plt.show()


# Experiment workflow:
# makeAllData(): Generate and save the training, test, and long-horizon datasets for this problem.
# doCrossValidationClassic(): Run cross-validation for the classical value-function surrogate and choose the kernel gamma.
# doCrossValidationControl(): Run cross-validation for the direct feedback-control surrogate and choose the kernel gamma.
# computeErrorsControl(): Train the direct-control surrogate with greedy centers and save its error curves.
# computeErrorsClassic(): Train the classical surrogate with greedy centers and save its error curves.
# plotResult(): Load saved surrogates and plot greedy, test, and long-horizon performance errors.

# makeAllData()
# doCrossValidationClassic()
# doCrossValidationControl()
computeErrorsControl()
computeErrorsClassic()
plotResult()


