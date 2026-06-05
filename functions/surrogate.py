"""Surrogate classes for classical value-function approximation and direct feedback-control approximation."""

import numpy as np
import pickle
from scipy.integrate import solve_ivp


class SurrogateControl:
    """Surrogate for direct approximation of the feedback control."""

    def __init__(self, kernel, model):
        """Initialize the object and store the required parameters."""
        self.trainError      = []
        self.testError       = []
        self.testErrorLongHo = []
        self.alpha           = []
        self.center          = []
        self.kernel          = kernel
        self.model           = model
        self.C               = np.array([])
        self.funcCenterList  = []
        self.usedRHS         = []

    def saveSr(self, name):
        """Save the trained surrogate in the data folder."""
        model_backup = self.model
        self.model   = None
        with open('data/' + name, 'wb') as outp:
            pickle.dump(self, outp, protocol=4)
        self.model = model_backup

    def loadSr(self, name, model=None):
        """Load a stored surrogate and restore its coefficients, centers, and error histories."""
        current_model = self.model if model is None else model
        with open('data/' + name, 'rb') as inp:
            oldSelf = pickle.load(inp)
        self.trainError      = oldSelf.trainError
        self.testError       = oldSelf.testError
        self.testErrorLongHo = oldSelf.testErrorLongHo
        self.alpha           = oldSelf.alpha
        self.center          = oldSelf.center
        self.kernel          = oldSelf.kernel
        self.C               = oldSelf.C
        self.funcCenterList  = oldSelf.funcCenterList
        self.usedRHS         = oldSelf.usedRHS
        if oldSelf.model is None:
            self.model = current_model
        else:
            self.model = oldSelf.model

    def makeControlData(self, data):
        """Convert state and costate data into direct control interpolation data."""
        S    = data.state
        P    = data.costate
        d, M = S.shape
        U    = []
        for j in range(M):
            u = self.model.getMuFromSr(S[:, j], P[:, j])
            u = np.asarray(u, dtype=float).reshape(-1)
            U.append(u)
        U         = np.column_stack(U)
        m         = U.shape[0]
        Y         = np.tile(S, (1, m))
        funcYList = np.repeat(np.arange(1, m + 1), M)
        rhs       = U.reshape(-1)
        return (Y, funcYList, rhs)

    def doFGreedy(self, data, nMaxGreedy, eps=10 ** (-16)):
        """Select greedy centers, fit the surrogate iteratively, and store the training error."""
        self.C              = np.array([])
        Y, funcYList, rhs   = self.makeControlData(data)
        KYX                 = np.zeros((Y.shape[1], nMaxGreedy))
        idx                 = np.argmax(np.abs(rhs))
        self.center         = np.atleast_2d(Y[:, idx]).T
        self.funcCenterList = np.atleast_1d(funcYList[idx])
        self.trainError     = np.atleast_1d(np.abs(rhs[idx]))
        self.usedRHS        = np.atleast_1d(rhs[idx])
        i                   = 0
        while i < nMaxGreedy and self.trainError[-1] > eps:
            self.fit()
            KYX[:, i]           = self.kernel.getGramControl(Y, funcYList, np.atleast_2d(self.center[:, i]).T, np.atleast_1d(self.funcCenterList[i]), self.model.controlWeight, self.model.g)[:, 0]
            preOut              = KYX[:, :i + 1] @ self.alpha
            res                 = np.abs(preOut - rhs)
            idx                 = np.argmax(res)
            self.trainError     = np.r_[self.trainError, res[idx]]
            self.center         = np.c_[self.center, np.atleast_2d(Y[:, idx]).T]
            self.funcCenterList = np.r_[self.funcCenterList, funcYList[idx]]
            self.usedRHS        = np.r_[self.usedRHS, rhs[idx]]
            print(str(i) + ' iteration steps with a control training error of ' + str(self.trainError[-1]))
            i += 1
        self.fit()

    def fit(self, iterativ=True):
        """Solve or update the interpolation system for the current surrogate centers."""
        if iterativ:
            if self.center.shape[1] == 1:
                K          = self.kernel.getGramControl(self.center, self.funcCenterList, self.center, self.funcCenterList, self.model.controlWeight, self.model.g)
                self.alpha = np.linalg.solve(K, self.usedRHS)
                L          = np.linalg.cholesky(K)
                self.C     = np.linalg.solve(L, np.eye(L.shape[0]))
            else:
                KX       = self.kernel.getGramControl(self.center[:, :-1], self.funcCenterList[:-1], np.atleast_2d(self.center[:, -1]).T, np.atleast_1d(self.funcCenterList[-1]), self.model.controlWeight, self.model.g)
                KXX      = self.kernel.getGramControl(np.atleast_2d(self.center[:, -1]).T, np.atleast_1d(self.funcCenterList[-1]), np.atleast_2d(self.center[:, -1]).T, np.atleast_1d(self.funcCenterList[-1]), self.model.controlWeight, self.model.g)
                dm       = self.C @ KX
                radicand = KXX - dm.T @ dm
                if np.any(radicand <= 0):
                    print('Warning: non-positive Cholesky update:', radicand)
                    radicand = np.maximum(radicand, 1e-300)
                dmm        = np.sqrt(radicand)
                cmm        = 1 / dmm
                cm         = -self.C.T @ (dm @ cmm.T)
                self.C     = np.c_[np.r_[self.C, cm.T], np.r_[cm * 0, cmm]]
                self.alpha = self.C.T @ (self.C @ self.usedRHS)
        else:
            K          = self.kernel.getGramControl(self.center, self.funcCenterList, self.center, self.funcCenterList, self.model.controlWeight, self.model.g)
            self.alpha = np.linalg.solve(K, self.usedRHS)

    def doFGreedyWithError(self, endTLarg, data, nMaxGreedy, dataTest, dataTestLongHo, eps=10 ** (-16)):
        """Train the surrogate with greedy selection and record training, test, and long-horizon errors."""
        self.C                        = np.array([])
        Y, funcYList, rhs             = self.makeControlData(data)
        YTest, funcYListTest, rhsTest = self.makeControlData(dataTest)
        KYX                           = np.zeros((Y.shape[1], nMaxGreedy))
        KYXTest                       = np.zeros((YTest.shape[1], nMaxGreedy))
        idx                           = np.argmax(np.abs(rhs))
        self.center                   = np.atleast_2d(Y[:, idx]).T
        self.funcCenterList           = np.atleast_1d(funcYList[idx])
        self.trainError               = np.atleast_1d(np.abs(rhs[idx]))
        self.usedRHS                  = np.atleast_1d(rhs[idx])
        self.testError                = np.atleast_1d(np.max(np.abs(rhsTest)))
        self.testErrorLongHo          = 0
        i                             = 0
        while i < nMaxGreedy and self.trainError[-1] > eps:
            self.fit()
            costCum = 0.0
            if i > -1:
                for j in range(dataTestLongHo.state.shape[1]):
                    _, _, cost = self.solveClosedLoopControl(dataTestLongHo.state[:, j], endTLarg)
                    costCum    = max(costCum, abs(cost - dataTestLongHo.vf[0, j]))
                self.testErrorLongHo = np.r_[self.testErrorLongHo, np.atleast_1d(costCum)]
            KYX[:, i]           = self.kernel.getGramControl(Y, funcYList, np.atleast_2d(self.center[:, i]).T, np.atleast_1d(self.funcCenterList[i]), self.model.controlWeight, self.model.g)[:, 0]
            KYXTest[:, i]       = self.kernel.getGramControl(YTest, funcYListTest, np.atleast_2d(self.center[:, i]).T, np.atleast_1d(self.funcCenterList[i]), self.model.controlWeight, self.model.g)[:, 0]
            preOut              = KYX[:, :i + 1] @ self.alpha
            res                 = np.abs(preOut - rhs)
            idx                 = np.argmax(res)
            self.trainError     = np.r_[self.trainError, res[idx]]
            self.center         = np.c_[self.center, np.atleast_2d(Y[:, idx]).T]
            self.funcCenterList = np.r_[self.funcCenterList, funcYList[idx]]
            self.usedRHS        = np.r_[self.usedRHS, rhs[idx]]
            preOutTest          = KYXTest[:, :i + 1] @ self.alpha
            resTest             = np.abs(preOutTest - rhsTest)
            self.testError      = np.r_[self.testError, np.max(resTest)]
            print(str(i) + ' iteration steps with a control training error of ' + str(self.trainError[-1]) + ', test error ' + str(self.testError[-1]) + ', long-horizon error ' + str(costCum))
            i += 1
        self.fit()
        self.testErrorLongHo = self.testErrorLongHo[1:]

    def evalGen(self, y, funcYList):
        """Evaluate the surrogate for general functionals."""
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return self.kernel.getGramControl(y, funcYList, self.center, self.funcCenterList, self.model.controlWeight, self.model.g) @ self.alpha

    def evalSurrogateControl(self, y):
        """Evaluate the direct feedback-control surrogate at one or more states."""
        y           = np.asarray(y, dtype=float)
        vectorInput = y.ndim == 1
        if vectorInput:
            y = y.reshape(-1, 1)
        d, M        = y.shape
        zeroCostate = np.zeros(d)
        u0          = self.model.getMuFromSr(y[:, 0], zeroCostate)
        m           = np.asarray(u0).reshape(-1).shape[0]
        YEval       = np.tile(y, (1, m))
        funcYList   = np.repeat(np.arange(1, m + 1), M)
        values      = self.evalGen(YEval, funcYList)
        U           = values.reshape(m, M)
        if vectorInput:
            return U[:, 0]
        return U

    def solveClosedLoopControl(self, startState, endT):
        """Simulate the closed-loop system with the directly approximated control and compute the cost."""
        x0 = np.asarray(startState, dtype=float).reshape(-1)

        def rhs(t, y_aug):
            """Evaluate the right-hand side used by the current simulation."""
            x  = y_aug[:-1]
            u  = self.evalSurrogateControl(x)
            dx = self.model.getF(x, u)
            dx = np.asarray(dx).reshape(-1)
            dJ = -float(np.asarray(self.model.getRHS(x, u)).reshape(-1)[0])
            return np.concatenate([dx, [dJ]])
        y0   = np.concatenate([x0, [0.0]])
        sol  = solve_ivp(rhs, (0.0, float(endT)), y0, method='DOP853', rtol=1e-10, atol=1e-10, max_step=1e+16)
        xT   = sol.y[:-1, -1]
        cost = sol.y[-1, -1] + self.model.terminalCost(xT)
        return (sol.y[:-1], sol.t, cost)


class SurrogateClassic:
    """Classical surrogate for the value function and its gradient."""

    def __init__(self, kernel):
        """Initialize the object and store the required parameters."""
        self.trainError      = []
        self.testError       = []
        self.testErrorLongHo = []
        self.alpha           = []
        self.center          = []
        self.kernel          = kernel
        self.C               = np.array([])

    def saveSr(self, name):
        """Save the trained surrogate in the data folder."""
        with open('data/' + name, 'wb') as outp:
            pickle.dump(self, outp, protocol=4)

    def loadSr(self, name):
        """Load a stored surrogate and restore its coefficients, centers, and error histories."""
        with open('data/' + name, 'rb') as inp:
            oldSelf              = pickle.load(inp)
            self.trainError      = oldSelf.trainError
            self.alpha           = oldSelf.alpha
            self.center          = oldSelf.center
            self.kernel          = oldSelf.kernel
            self.C               = oldSelf.C
            self.funcCenterList  = oldSelf.funcCenterList
            self.usedRHS         = oldSelf.usedRHS
            self.testError       = oldSelf.testError
            self.testErrorLongHo = oldSelf.testErrorLongHo

    def doFGreedy(self, data, nMaxGreedy, eps=10 ** (-16)):
        """Select greedy centers, fit the surrogate iteratively, and store the training error."""
        self.C              = np.array([])
        S                   = data.state
        N, M                = S.shape
        Y                   = np.tile(S, (1, N + 1))
        funcYList           = np.repeat(np.arange(N + 1), M).astype(int)
        rhs                 = np.r_[data.vf[0, :], data.costate.flatten()]
        KYX                 = np.zeros((Y.shape[1], nMaxGreedy))
        idx                 = np.argmax(np.abs(rhs[:M]))
        self.center         = np.atleast_2d(Y[:, idx]).T
        self.funcCenterList = np.atleast_1d(funcYList[idx])
        self.trainError     = np.atleast_1d(np.abs(rhs[idx]))
        self.usedRHS        = np.atleast_1d(rhs[idx])
        i                   = 0
        while i < nMaxGreedy and self.trainError[-1] > eps:
            self.fit()
            KYX[:, i]           = self.kernel.getGramHermite(Y, funcYList, np.atleast_2d(self.center[:, i]).T, np.atleast_1d(self.funcCenterList[i]))[:, 0]
            preOut              = KYX[:, :i + 1] @ self.alpha
            res                 = np.abs(preOut - rhs)
            idx                 = np.argmax(res[:M])
            self.trainError     = np.r_[self.trainError, res[idx]]
            self.center         = np.c_[self.center, np.atleast_2d(Y[:, idx]).T]
            self.funcCenterList = np.r_[self.funcCenterList, funcYList[idx]]
            self.usedRHS        = np.r_[self.usedRHS, rhs[idx]]
            print(str(i) + ' iteration steps with  an training error of ' + str(self.trainError[-1]))
            i = i + 1
        self.fit()

    def fit(self, iterativ=True):
        """Solve or update the interpolation system for the current surrogate centers."""
        if iterativ:
            if self.center.shape[1] == 1:
                K          = self.kernel.getGramHermite(self.center, self.funcCenterList, self.center, self.funcCenterList)
                self.alpha = np.linalg.solve(K, self.usedRHS)
                L          = np.linalg.cholesky(K)
                self.C     = np.linalg.solve(L, np.eye(L.shape[0]))
            else:
                KX         = self.kernel.getGramHermite(self.center[:, :-1], self.funcCenterList[:-1], np.atleast_2d(self.center[:, -1]).T, np.atleast_1d(self.funcCenterList[-1]))
                KXX        = self.kernel.getGramHermite(np.atleast_2d(self.center[:, -1]).T, np.atleast_1d(self.funcCenterList[-1]), np.atleast_2d(self.center[:, -1]).T, np.atleast_1d(self.funcCenterList[-1]))
                dm         = self.C @ KX
                dmm        = np.sqrt(KXX - dm.T @ dm)
                cmm        = 1 / dmm
                cm         = -self.C.T @ (dm @ cmm.T)
                self.C     = np.c_[np.r_[self.C, cm.T], np.r_[cm * 0, cmm]]
                self.alpha = self.C.T @ (self.C @ self.usedRHS)
        else:
            K          = self.kernel.getGramHermite(self.center, self.funcCenterList, self.center, self.funcCenterList)
            self.alpha = np.linalg.solve(K, self.usedRHS)

    def doFGreedyWithError(self, model, endTLarg, data, nMaxGreedy, dataTest, dataTestLongHo, eps=10 ** (-16)):
        """Train the surrogate with greedy selection and record training, test, and long-horizon errors."""
        self.C               = np.array([])
        S                    = data.state
        N, M                 = S.shape
        Y                    = np.tile(S, (1, N + 1))
        funcYList            = np.repeat(np.arange(N + 1), M).astype(int)
        rhs                  = np.r_[data.vf[0, :], data.costate.flatten()]
        S                    = dataTest.state
        N, M2                = S.shape
        YTest                = np.tile(S, (1, N + 1))
        funcYListTest        = np.repeat(np.arange(N + 1), M2).astype(int)
        rhsTest              = np.r_[dataTest.vf[0, :], dataTest.costate.flatten()]
        idx                  = np.argmax(np.abs(rhsTest[:M2]))
        self.testError       = np.atleast_1d(np.abs(rhs[idx]))
        KYX                  = np.zeros((Y.shape[1], nMaxGreedy))
        KYXTest              = np.zeros((YTest.shape[1], nMaxGreedy))
        idx                  = np.argmax(np.abs(rhs[:M]))
        self.center          = np.atleast_2d(Y[:, idx]).T
        self.funcCenterList  = np.atleast_1d(funcYList[idx])
        self.trainError      = np.atleast_1d(np.abs(rhs[idx]))
        self.usedRHS         = np.atleast_1d(rhs[idx])
        i                    = 0
        self.testErrorLongHo = 0
        while i < nMaxGreedy and self.trainError[-1] > eps:
            self.fit()
            costCum = 0
            if i > -1:
                for j in range(dataTestLongHo.state.shape[1]):
                    _, _, cost = model.solveSurrogate(dataTestLongHo.state[:, j], endTLarg, self.evalGrad)
                    costCum    = np.max([costCum, np.abs(cost - dataTestLongHo.vf[0, j])])
            self.testErrorLongHo = np.r_[self.testErrorLongHo, np.atleast_1d(costCum)]
            KYX[:, i]            = self.kernel.getGramHermite(Y, funcYList, np.atleast_2d(self.center[:, i]).T, np.atleast_1d(self.funcCenterList[i]))[:, 0]
            KYXTest[:, i]        = self.kernel.getGramHermite(YTest, funcYListTest, np.atleast_2d(self.center[:, i]).T, np.atleast_1d(self.funcCenterList[i]))[:, 0]
            preOut              = KYX[:, :i + 1] @ self.alpha
            res                 = np.abs(preOut - rhs)
            idx                 = np.argmax(res[:M])
            self.trainError     = np.r_[self.trainError, res[idx]]
            self.center         = np.c_[self.center, np.atleast_2d(Y[:, idx]).T]
            self.funcCenterList = np.r_[self.funcCenterList, funcYList[idx]]
            self.usedRHS        = np.r_[self.usedRHS, rhs[idx]]
            preOut              = KYXTest[:, :i + 1] @ self.alpha
            res                 = np.abs(preOut - rhsTest)
            idx                 = np.argmax(res[:M2])
            self.testError      = np.r_[self.testError, res[idx]]
            print(str(i) + ' iteration steps with  an training error of ' + str(self.trainError[-1]) + ', test error ' + str(self.testError[-1]) + ', long-horizon error ' + str(self.testErrorLongHo[-1]))
            i = i + 1
        self.fit()
        self.testErrorLongHo = self.testErrorLongHo[1:]

    def evalGen(self, y, funcYList):
        """Evaluate the surrogate for general functionals."""
        return self.kernel.getGramHermite(y, funcYList, self.center, self.funcCenterList) @ self.alpha

    def evalFunc(self, y):
        """Evaluate the approximated value function."""
        return self.evalGen(self, y, [0])

    def evalGrad(self, y):
        """Evaluate the approximated gradient of the value function."""
        return self.evalGen(np.tile(y[:, None], (1, y.shape[0])), list(range(1, y.shape[0] + 1)))
