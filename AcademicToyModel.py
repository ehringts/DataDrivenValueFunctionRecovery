import numpy             as np
import matplotlib.pyplot as plt
from   matplotlib.ticker import LogLocator, NullFormatter
from   functions         import  model,data,kernel, surrogate,auxFunctions

dim          = 4
newModel     = model.AcademicToyModelII(dim,1/2,1)
endT         = 10


def makeAllData():
    def make_grid(N, num=100, a=-0.5, b=0.5):
        x     = np.linspace(a, b, num)
        grids = np.meshgrid(*([x] * N), indexing="ij")   # N arrays, each shape (num,)*N
        omega = np.stack([g.ravel() for g in grids], axis=0)  # shape (N, num**N)
        return omega
    omega             = make_grid(dim,10,-0.49,0.49)
    newDataTrain      = data.TrainData()
    newDataTest       = data.TestData()
    newDataTestLongHo = data.TestData()
    newDataTrain.makeTrainData    (newModel,endT,endT/10,100,omega,0.1,10000,np.array([[-0.5,0.5],[-0.5,0.5],[-0.5,0.5],[-0.5,0.5]]),"trainAcademicToyModel2") # 7700 iteration steps with an fill distance of 0.09652049059559197
    newDataTest.makeTestData      (newModel,endT*10,endT/10,100,omega,100,1213,"testAcademicToyModel2")
    newDataTestLongHo.makeTestData(newModel,endT*10,endT/10,100,omega/2,10,151195,"testAcademicToyModel2LongHorizon")
    newDataTest.vf = np.atleast_2d(newModel.trueVF(newDataTest.state))
    newDataTest.save("testAcademicToyModel2")    
    newDataTestLongHo.vf = np.atleast_2d(newModel.trueVF(newDataTestLongHo.state))
    newDataTestLongHo.save("testAcademicToyModel2LongHorizon")
    

def doCrossValidationControl():
    newDataTrain      = data.TrainData("trainAcademicToyModel2")
    gammaList         = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

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


    #Best gamma for LinMatern: 0.03 with error: 0.05702933858992413
    #Best gamma for LinMaternProduct: 0.04 with error: 0.019835022281989096
    #Best gamma for QuadMatern: 0.01 with error: 0.022222779092358685
    #Best gamma for QuadMaternProduct: 0.1 with error: 0.007560855714431399
    #Best gamma for CubMatern: 0.02 with error: 0.010240496644050412
    #Best gamma for CubMaternProduct: 0.1 with error: 0.0060322794203732655 !!!!!!!!!!!!!!!!!!!!
    #Best gamma for QuartMatern: 0.09 with error: 0.0059071366292418786
    #Best gamma for QuartMaternProduct: 0.4 with error: 0.008413550827859292


def doCrossValidationClassic():
    newDataTrain      = data.TrainData("trainAcademicToyModel2")
    gammaList         = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

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

    #Best gamma for LinMatern: 0.01 with error: 0.029458292877249104
    #Best gamma for LinMaternProduct: 0.06 with error: 0.010105704061274867
    #Best gamma for QuadMatern: 0.06 with error: 0.01400652478113377
    #Best gamma for QuadMaternProduct: 0.04 with error: 0.00467824757451026
    #Best gamma for CubMatern: 0.3 with error: 0.00802979502379427
    #Best gamma for CubMaternProduct: 0.1 with error: 0.0026396204359194395
    #Best gamma for QuartMatern: 0.3 with error: 0.0050911961010205635
    #Best gamma for QuartMaternProduct: 0.3 with error: 0.003521211768880317


def computeErrorsControl():
    newkernel         = kernel.CubMaternProduct(0.1)
    newSurrogate      = surrogate.SurrogateControl(newkernel, newModel)
    newDataTrain      = data.TrainData("trainAcademicToyModel2")
    newDataTest       = data.TestData("testAcademicToyModel2")
    newDataTestLongHo = data.TestData("testAcademicToyModel2LongHorizon")
    newSurrogate.doFGreedyWithError(endT*10,newDataTrain,200,newDataTest,newDataTestLongHo)
    newSurrogate.saveSr("surrogateAcademicToyModel2Control")


def computeErrorsClassic():
    newkernel         = kernel.CubMaternProduct(0.1)
    newSurrogate      = surrogate.SurrogateClassic(newkernel)
    newDataTrain      = data.TrainData("trainAcademicToyModel2")
    newDataTest       = data.TestData("testAcademicToyModel2")
    newDataTestLongHo = data.TestData("testAcademicToyModel2LongHorizon")
    newSurrogate.doFGreedyWithError(newModel,endT*10,newDataTrain,200,newDataTest,newDataTestLongHo)
    newSurrogate.saveSr("surrogateAcademicToyModel2Classic")


def plotResult():
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
        newSurrogateControl.loadSr("surrogateAcademicToyModel2Control")

        newSurrogateClassic = surrogate.SurrogateClassic(newkernel)
        newSurrogateClassic.loadSr("surrogateAcademicToyModel2Classic")

        wid = 1.6
        fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)

        colors = {
            "Greedy-Error": "C0",
            "Test-Error": "C1",
            "Performance-Error": "C2",
        }


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

        ax.set_title(r"Academic Toy Model (\texttt{AcademicToy})", pad=10)

        ax.set_xlim(1, min(200, xmax))

        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
        ax.yaxis.set_minor_formatter(NullFormatter())

        ax.grid(True, which="major", alpha=0.6, linewidth=0.8)
        #ax.grid(True, which="minor", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)

        # --- Save MAIN plot WITHOUT legend ---
        fig.savefig("AcademicToyModel.pdf", bbox_inches="tight")

        # --- Separate figure ONLY for legend ---
        handles, labels = ax.get_legend_handles_labels()

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




# computeErrorsControl()

# computeErrorsClassic()

plotResult()