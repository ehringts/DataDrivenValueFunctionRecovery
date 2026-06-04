import numpy             as np
import matplotlib.pyplot as plt
from   matplotlib.ticker import LogLocator, NullFormatter
from   functions         import  model,data,kernel, surrogate,auxFunctions



newModel = model.CoupledDuffingOscillator4D()
endT     = 10


def makeAllData():
    x = np.linspace(-1, 1, 40)
    X, Y, Z, W = np.meshgrid(x, x, x, x)

    omega = np.c_[
        X.flatten(),
        Y.flatten(),
        Z.flatten(),
        W.flatten(),
    ].T

    cubeBound = np.array([
        [-1, 1],
        [-1, 1],
        [-1, 1],
        [-1, 1]
    ])
    newDataTrain      = data.TrainData()
    newDataTest       = data.TestData()
    newDataTestLongHo = data.TestData()
    newDataTrain.makeTrainData    (newModel,endT,endT/10,100,omega,0.2,10000,cubeBound,saveName="trainDataCoupledDuffing",BVP=True)
    ##### newDataTest.makeTestData      (newModel,endT*10,endT/10,100,omega,100,5421484,"testDataCoupledDuffing",BVP=True)
    ##### Do Not Touch newDataTestLongHo.makeTestData(newModel,endT*10,endT/10,100,omega/2,10,9441,"testDataCoupledDuffingLongHorizon",BVP=True)

def doCrossValidationControl():
    newDataTrain      = data.TrainData("trainDataCoupledDuffing")
    gammaList         = [0.05,0.06,0.07,0.08,0.09,0.1,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2]
    #gammaList         = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]

    nFolds            = 3
    iterMax           = 200

    newSurrogate      = surrogate.SurrogateControl(kernel.LinMatern(1.29), newModel)
    minGamm1, min1 ,_ = auxFunctions.crossValidationControlList(newDataTrain, newSurrogate, nFolds, iterMax,gammaList)
    newSurrogate      = surrogate.SurrogateControl(kernel.LinMaternProduct(1.29), newModel)
    minGamm2, min2 ,_ = auxFunctions.crossValidationControlList(newDataTrain, newSurrogate, nFolds, iterMax,gammaList)    
    newSurrogate      = surrogate.SurrogateControl(kernel.QuadMatern(1.29), newModel)
    minGamm3, min3 ,_ = auxFunctions.crossValidationControlList(newDataTrain, newSurrogate, nFolds, iterMax,gammaList)
    newSurrogate      = surrogate.SurrogateControl(kernel.QuadMaternProduct(1.29), newModel)
    minGamm4, min4 ,_ = auxFunctions.crossValidationControlList(newDataTrain, newSurrogate, nFolds, iterMax,gammaList) 
    newSurrogate      = surrogate.SurrogateControl(kernel.CubMatern(1.29), newModel)
    minGamm5, min5 ,_ = auxFunctions.crossValidationControlList(newDataTrain, newSurrogate, nFolds, iterMax,gammaList)
    newSurrogate      = surrogate.SurrogateControl(kernel.CubMaternProduct(1.29), newModel)
    minGamm6, min6 ,_ = auxFunctions.crossValidationControlList(newDataTrain, newSurrogate, nFolds, iterMax,gammaList) 
    newSurrogate      = surrogate.SurrogateControl(kernel.QuartMatern(1.29), newModel)
    minGamm7, min7 ,_ = auxFunctions.crossValidationControlList(newDataTrain, newSurrogate, nFolds, iterMax,gammaList)
    newSurrogate      = surrogate.SurrogateControl(kernel.QuartMaternProduct(1.29), newModel)
    minGamm8, min8 ,_ = auxFunctions.crossValidationControlList(newDataTrain, newSurrogate, nFolds, iterMax,gammaList)     

    print("Best gamma for LinMatern: " + str(minGamm1) + " with error: " + str(min1))
    print("Best gamma for LinMaternProduct: " + str(minGamm2) + " with error: " + str(min2))
    print("Best gamma for QuadMatern: " + str(minGamm3) + " with error: " + str(min3))
    print("Best gamma for QuadMaternProduct: " + str(minGamm4) + " with error: " + str(min4))
    print("Best gamma for CubMatern: " + str(minGamm5) + " with error: " + str(min5))
    print("Best gamma for CubMaternProduct: " + str(minGamm6) + " with error: " + str(min6))
    print("Best gamma for QuartMatern: " + str(minGamm7) + " with error: " + str(min7))
    print("Best gamma for QuartMaternProduct: " + str(minGamm8) + " with error: " + str(min8))



    #Best gamma for LinMatern: 0.05 with error: 0.2821091606179757
    #Best gamma for LinMaternProduct: 0.05 with error: 0.001790307504760058
    #Best gamma for QuadMatern: 0.05 with error: 0.006388096092772244
    #Best gamma for QuadMaternProduct: 0.08 with error: 0.00025367404144815975
    #Best gamma for CubMatern: 0.05 with error: 0.00040238197677772147
    #Best gamma for CubMaternProduct: 0.1 with error: 0.00018599107332158601
    #Best gamma for QuartMatern: 0.1 with error: 0.0002436653531357654
    #Best gamma for QuartMaternProduct: 0.07 with error: 0.00010106901629873082

def doCrossValidationClassic():
    newDataTrain      = data.TrainData("trainDataCoupledDuffing")
    gammaList         = [0.05,0.06,0.07,0.08,0.09,0.1,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3]
    #gammaList         = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]

    nFolds            = 3
    iterMax           = 200

    newSurrogate      = surrogate.SurrogateClassic(kernel.LinMatern(1.29))
    minGamm1, min1 ,_ = auxFunctions.crossValidationListClassic(newDataTrain, newSurrogate, nFolds, iterMax,gammaList)
    newSurrogate      = surrogate.SurrogateClassic(kernel.LinMaternProduct(1.29))
    minGamm2, min2 ,_ = auxFunctions.crossValidationListClassic(newDataTrain, newSurrogate, nFolds, iterMax,gammaList)    
    newSurrogate      = surrogate.SurrogateClassic(kernel.QuadMatern(1.29))
    minGamm3, min3 ,_ = auxFunctions.crossValidationListClassic(newDataTrain, newSurrogate, nFolds, iterMax,gammaList)
    newSurrogate      = surrogate.SurrogateClassic(kernel.QuadMaternProduct(1.29))
    minGamm4, min4 ,_ = auxFunctions.crossValidationListClassic(newDataTrain, newSurrogate, nFolds, iterMax,gammaList) 
    newSurrogate      = surrogate.SurrogateClassic(kernel.CubMatern(1.29))
    minGamm5, min5 ,_ = auxFunctions.crossValidationListClassic(newDataTrain, newSurrogate, nFolds, iterMax,gammaList)
    newSurrogate      = surrogate.SurrogateClassic(kernel.CubMaternProduct(1.29))
    minGamm6, min6 ,_ = auxFunctions.crossValidationListClassic(newDataTrain, newSurrogate, nFolds, iterMax,gammaList) 
    newSurrogate      = surrogate.SurrogateClassic(kernel.QuartMatern(1.29))
    minGamm7, min7 ,_ = auxFunctions.crossValidationListClassic(newDataTrain, newSurrogate, nFolds, iterMax,gammaList)
    newSurrogate      = surrogate.SurrogateClassic(kernel.QuartMaternProduct(1.29))
    minGamm8, min8 ,_ = auxFunctions.crossValidationListClassic(newDataTrain, newSurrogate, nFolds, iterMax,gammaList)     

    print("Best gamma for LinMatern: " + str(minGamm1) + " with error: " + str(min1))
    print("Best gamma for LinMaternProduct: " + str(minGamm2) + " with error: " + str(min2))
    print("Best gamma for QuadMatern: " + str(minGamm3) + " with error: " + str(min3))
    print("Best gamma for QuadMaternProduct: " + str(minGamm4) + " with error: " + str(min4))
    print("Best gamma for CubMatern: " + str(minGamm5) + " with error: " + str(min5))
    print("Best gamma for CubMaternProduct: " + str(minGamm6) + " with error: " + str(min6))
    print("Best gamma for QuartMatern: " + str(minGamm7) + " with error: " + str(min7))
    print("Best gamma for QuartMaternProduct: " + str(minGamm8) + " with error: " + str(min8))


    #Best gamma for LinMatern: 0.07 with error: 0.027153267104802037
    #Best gamma for LinMaternProduct: 0.05 with error: 0.0010007975955207786
    #Best gamma for QuadMatern: 0.06 with error: 0.0020229626842651616
    #Best gamma for QuadMaternProduct: 0.05 with error: 6.606195264213799e-05
    #Best gamma for CubMatern: 0.07 with error: 0.0004099087254268811
    #Best gamma for CubMaternProduct: 0.07 with error: 1.8715804819565324e-05
    #Best gamma for QuartMatern: 0.3 with error: 0.0007025654707346618
    #Best gamma for QuartMaternProduct: 0.07 with error: 1.5049650095022571e-05


# 7806 iteration steps with an fill distance of 0.19993298064830964
def computeErrorsControl():
    newkernel         = kernel.QuartMaternProduct(0.07) # 0.1 was the best
    newSurrogate      = surrogate.SurrogateControl(newkernel,newModel)
    newDataTrain      = data.TrainData("trainDataCoupledDuffing")
    newDataTest       = data.TestData("testDataCoupledDuffing")
    newDataTestLongHo = data.TestData("testDataCoupledDuffingLongHorizon")
    newSurrogate.doFGreedyWithError(endT*10,newDataTrain,200,newDataTest,newDataTestLongHo)
    newSurrogate.saveSr("surrogateCoupledDuffingControl")

def computeErrorsClassic():
    newkernel         = kernel.QuartMaternProduct(0.07)  # 0.1 was the best
    newSurrogate      = surrogate.SurrogateClassic(newkernel)
    newDataTrain      = data.TrainData("trainDataCoupledDuffing")
    newDataTest       = data.TestData("testDataCoupledDuffing")
    newDataTestLongHo = data.TestData("testDataCoupledDuffingLongHorizon")
    newSurrogate.doFGreedyWithError(newModel,endT*10,newDataTrain,200,newDataTest,newDataTestLongHo)
    newSurrogate.saveSr("surrogateCoupledDuffingClassic")






def plotResult():
    # --- LaTeX look + bigger typography (requires a LaTeX installation) ---
    rc = {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "lines.linewidth": 1.6,
        "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
    }

    with plt.rc_context(rc):
        newkernel = kernel.GaussProduct(1.07)

        newSurrogateControl = surrogate.SurrogateControl(newkernel, newModel)
        newSurrogateControl.loadSr("surrogateCoupledDuffingControl")

        newSurrogateClassic = surrogate.SurrogateClassic(newkernel)
        newSurrogateClassic.loadSr("surrogateCoupledDuffingClassic")

        wid = 1.6
        fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)

        colors = {
            "Greedy-Error": "C0",
            "Test-Error": "C1",
            "Performance-Error": "C2",
        }

        # Desired indexing:
        #
        # trainError:
        #   ignore y[0]
        #   plot y[1] at x = 1
        #   plot y[2] at x = 2
        #   ...
        #
        # testError:
        #   ignore y[0]
        #   plot y[1] at x = 1
        #   plot y[2] at x = 2
        #   ...
        #
        # testErrorLongHo:
        #   do NOT ignore first value
        #   plot y[0] at x = 1
        #   plot y[1] at x = 2
        #   ...

        series = [
            # metric,              data,                               linestyle, method,      x_start, drop_first
            ("Greedy-Error",      newSurrogateClassic.trainError,      "-",      "Classical", 1,       True),
            ("Test-Error",        newSurrogateClassic.testError,       "-",      "Classical", 1,       True),
            ("Performance-Error", newSurrogateClassic.testErrorLongHo, "-",      "Classical", 1,       False),

            ("Greedy-Error",      newSurrogateControl.trainError,      "--",     "Control",   1,       True),
            ("Test-Error",        newSurrogateControl.testError,       "--",     "Control",   1,       True),
            ("Performance-Error", newSurrogateControl.testErrorLongHo, "--",     "Control",   1,       False),
        ]

        xmax = 1

        for metric, y, ls, method, x_start, drop_first in series:
            y = np.asarray(y)

            if drop_first:
                y = y[1:]

            if len(y) == 0:
                continue

            x = np.arange(x_start, x_start + len(y))
            xmax = max(xmax, x[-1])

            ax.plot(
                x,
                y,
                lw=wid,
                ls=ls,
                color=colors[metric],
                label=rf"{metric} ({method})"
            )

        ax.set_yscale("log")

        # Optional axis labels
        # ax.set_xlabel(r"Number of centers")
        # ax.set_ylabel(r"Error")

        ax.set_title(r"Coupled Duffing Oscillator (\texttt{CoupledDuffing})", pad=10)

        ax.set_xlim(1, min(200, xmax))

        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
        ax.yaxis.set_minor_formatter(NullFormatter())

        ax.grid(True, which="major", alpha=0.6, linewidth=0.8)
        #ax.grid(True, which="minor", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)

        # --- Save MAIN plot WITHOUT legend ---
        fig.savefig("CoupledDuffing.pdf", bbox_inches="tight")

        # --- Separate figure ONLY for legend ---
        handles, labels = ax.get_legend_handles_labels()

        # Current logical order:
        # 0 Classical Greedy
        # 1 Classical Test
        # 2 Classical Performance
        # 3 Control Greedy
        # 4 Control Test
        # 5 Control Performance
        #
        # Matplotlib fills legends column-wise when using ncol=3.
        # Therefore this reordered list gives visual rows:
        #
        # Row 1: Classical Greedy | Classical Test | Classical Performance
        # Row 2: Control Greedy   | Control Test   | Control Performance

        legend_order = [0, 3, 1, 4, 2, 5]

        handles = [handles[i] for i in legend_order]
        labels = [labels[i] for i in legend_order]

        fig_leg, ax_leg = plt.subplots(figsize=(9, 1.3), constrained_layout=True)
        ax_leg.axis("off")

        ax_leg.legend(
            handles,
            labels,
            loc="center",
            ncol=3,
            frameon=False,
            handlelength=3.0,
            columnspacing=1.4,
            handletextpad=0.6,
            borderaxespad=0.0,
        )

        # --- Draw a box around the whole legend figure ---
        fig_leg.patch.set_facecolor("white")
        fig_leg.patch.set_edgecolor("black")
        fig_leg.patch.set_linewidth(1.2)

        fig_leg.savefig("legend.pdf", bbox_inches="tight", pad_inches=0.05)

        plt.show()

# makeAllData()
# doCrossValidationControl()
# doCrossValidationClassic()

# computeErrorsControl()
# computeErrorsClassic()

plotResult()