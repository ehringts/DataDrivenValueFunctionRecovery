import numpy             as np
import matplotlib.pyplot as plt
from   matplotlib.ticker import LogLocator, NullFormatter
from   functions         import  model,data,kernel, surrogate,auxFunctions
import pickle



N    = 99
r    = 8
endT = 5
with open("data/ROMBasis.pkl", "rb") as f:
    PhiPOD = pickle.load(f)

np.save("data/PhiPOD.npy", PhiPOD)

newModel  = model.HEDirichletROM(N, [1/4,1,100,1],PhiPOD)



def makeAllData():
    def make_grid(N, num=100, a=-0.5, b=0.5):
        x     = np.linspace(a, b, num)
        grids = np.meshgrid(*([x] * N), indexing="ij")   # N arrays, each shape (num,)*N
        omega = np.stack([g.ravel() for g in grids], axis=0)  # shape (N, num**N)
        return omega

    omega =  make_grid(r,5,a=-2, b=2)

    cubeBound = np.array([
        [-2, 2],
        [-2, 2],
        [-2, 2],
        [-2, 2],
        [-2, 2],
        [-2, 2],
        [-2, 2],
        [-2, 2]
    ])
    newDataTrain      = data.TrainData()
    newDataTest       = data.TestData()
    newDataTestLongHo = data.TestData()
    newDataTrain.makeTrainData    (newModel,endT,endT/10,100,omega,2,10000,cubeBound,saveName="trainDataHeatROM",BVP=True)
    newDataTest.makeTestData      (newModel,endT*10,endT/10,100,omega,100,5421484,"testDataHeatROM",BVP=True)
    newDataTestLongHo.makeTestData(newModel,endT*10,endT/10,100,omega/2,10,9441,"testDataHeatROMLongHorizon",BVP=True)

def doCrossValidationControl():
    newDataTrain      = data.TrainData("trainDataHeatROM")
    gammaList         = [0.05,0.06,0.07,0.08,0.09,0.1,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2]
    nFolds            = 3
    iterMax           = 500

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


    #Best gamma for LinMatern: 0.05 with error: 0.048919184024094155
    #Best gamma for LinMaternProduct: 0.05 with error: 0.0033846160276923
    #Best gamma for QuadMatern: 0.05 with error: 0.006537347417824783
    #Best gamma for QuadMaternProduct: 0.06 with error: 0.0015852617781748068
    #Best gamma for CubMatern: 0.06 with error: 0.002148220243651707
    #Best gamma for CubMaternProduct: 0.07 with error: 0.0014239319254422216
    #Best gamma for QuartMatern: 0.07 with error: 0.0014206052893057282
    #Best gamma for QuartMaternProduct: 0.09 with error: 0.0016462140219188164

def doCrossValidationClassic():
    newDataTrain      = data.TrainData("trainDataHeatROM")
    gammaList         = [0.05,0.06,0.07,0.08,0.09,0.1,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3]
    #gammaList         = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]

    nFolds            = 3
    iterMax           = 500

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

    #Best gamma for LinMatern: 0.06 with error: 0.051093866203702286
    #Best gamma for LinMaternProduct: 0.05 with error: 0.0037646806989647765
    #Best gamma for QuadMatern: 0.05 with error: 0.009755644826290913
    #Best gamma for QuadMaternProduct: 0.06 with error: 0.001959273441105469
    #Best gamma for CubMatern: 0.05 with error: 0.003207426894373089
    #Best gamma for CubMaternProduct: 0.1 with error: 0.0017680542339456966
    #Best gamma for QuartMatern: 0.07 with error: 0.0017543395551901468
    #Best gamma for QuartMaternProduct: 0.1 with error: 0.0017056613125443671

def computeErrorsControl():
    newkernel         = kernel.CubMaternProduct(0.07)  
    newSurrogate      = surrogate.SurrogateControl(newkernel,newModel)
    newDataTrain      = data.TrainData("trainDataHeatROM")
    newDataTest       = data.TestData("testDataHeatROM")
    newDataTestLongHo = data.TestData("testDataHeatROMLongHorizon")
    newSurrogate.doFGreedyWithError(endT*10,newDataTrain,500,newDataTest,newDataTestLongHo)
    newSurrogate.saveSr("surrogateHeatROMControl")

def computeErrorsClassic():
    newkernel         = kernel.QuartMaternProduct(0.1)   
    newSurrogate      = surrogate.SurrogateClassic(newkernel)
    newDataTrain      = data.TrainData("trainDataHeatROM")
    newDataTest       = data.TestData("testDataHeatROM")
    newDataTestLongHo = data.TestData("testDataHeatROMLongHorizon")
    newSurrogate.doFGreedyWithError(newModel,endT*10,newDataTrain,500,newDataTest,newDataTestLongHo)
    newSurrogate.saveSr("surrogateHeatROMClassic")



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
        newSurrogateControl.loadSr("surrogateHeatROMControl")

        newSurrogateClassic = surrogate.SurrogateClassic(newkernel)
        newSurrogateClassic.loadSr("surrogateHeatROMClassic")

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

        ax.set_title(r"Reduced order heat equation (\texttt{ROMHeat})", pad=10)

        ax.set_xlim(1, min(500, xmax))

        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
        ax.yaxis.set_minor_formatter(NullFormatter())

        ax.grid(True, which="major", alpha=0.6, linewidth=0.8)
        #ax.grid(True, which="minor", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)

        # --- Save MAIN plot WITHOUT legend ---
        fig.savefig("HeatEquation.pdf", bbox_inches="tight")

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





#makeAllData()
#doCrossValidationControl()
#doCrossValidationClassic()

computeErrorsControl()
computeErrorsClassic()

plotResult()
