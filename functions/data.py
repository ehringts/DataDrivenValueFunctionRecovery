"""Dataset classes and routines for generating, saving, loading, and filtering training and test data."""

import numpy as np
import abc
import pickle


class Data(metaclass=abc.ABCMeta):
    """Abstract base class for datasets that can be saved to and loaded from the data folder."""

    def __init__(self, name=None):
        """Initialize the object and store the required parameters."""
        self.name = name
        if name is not None:
            self.load(name)

    @abc.abstractmethod
    def load(self):
        """Load stored data or a stored surrogate from the data folder."""

    @abc.abstractmethod
    def save(self):
        """Save the current data object in the data folder."""


class TrainData(Data):
    """Dataset class for training states, costates, and value-function data."""

    def __init__(self, name=None):
        """Initialize the object and store the required parameters."""
        super().__init__(name)

    def load(self, name):
        """Load stored data or a stored surrogate from the data folder."""
        with open('data/' + name, 'rb') as f:
            self.state, self.costate, self.vf = pickle.load(f)

    def save(self, name):
        """Save the current data object in the data folder."""
        with open('data/' + name, 'wb') as f:
            pickle.dump((self.state, self.costate, self.vf), f, protocol=4)

    def makeTrainData(self, model, T, deltaT, numbOfEval, omega, epsFD, nMaxFD, cubeBound, saveName=None, BVP=True):
        """Generate training data by iterative fill-distance sampling and save it optionally."""
        SOIS      = np.asarray(omega, dtype=float)
        cubeBound = np.asarray(cubeBound, dtype=float)
        if SOIS.ndim != 2:
            raise ValueError(f'omega must be two-dimensional with shape (d, nOmega). Got: {SOIS.shape}')
        if cubeBound.ndim != 2 or cubeBound.shape[1] != 2:
            raise ValueError(f'cubeBound must have shape (NCub, 2). Got: {cubeBound.shape}')
        d, nOmega = SOIS.shape
        NCub      = cubeBound.shape[0]

        def _mask_in_cube(X):
            """Return the mask of data points that lie inside the prescribed box."""
            if X.size == 0:
                return np.zeros((0,), dtype=bool)
            m    = X.shape[1]
            mask = np.ones(m, dtype=bool)
            for i in range(min(NCub, X.shape[0])):
                mask &= (X[i, :] >= cubeBound[i, 0]) & (X[i, :] <= cubeBound[i, 1])
            return mask

        def _apply_cube_filter(state, costate, vf):
            """Apply the box filter consistently to states, costates, and values."""
            mask = _mask_in_cube(state)
            return (state[:, mask], costate[:, mask], vf[:, mask])
        try:
            from scipy.spatial import cKDTree

            def _fill_distance_sq(SOIS, X):
                """Compute the squared fill distance between candidate points and existing data."""
                if X is None or X.shape[1] == 0:
                    norms2     = np.sum(SOIS * SOIS, axis=0)
                    idx        = int(np.argmax(norms2))
                    minList_sq = np.full(SOIS.shape[1], np.inf, dtype=float)
                    return (float(np.inf), idx, minList_sq)
                tree       = cKDTree(X.T)
                dist       = tree.query(SOIS.T, k=1, workers=-1)[0]
                minList_sq = dist * dist
                idx        = int(np.argmax(minList_sq))
                return (float(minList_sq[idx]), idx, minList_sq)
        except Exception:

            def _fill_distance_sq(SOIS, X, block=2048):
                """Compute the squared fill distance between candidate points and existing data."""
                if X is None or X.shape[1] == 0:
                    norms2     = np.sum(SOIS * SOIS, axis=0)
                    idx        = int(np.argmax(norms2))
                    minList_sq = np.full(SOIS.shape[1], np.inf, dtype=float)
                    return (float(np.inf), idx, minList_sq)
                nOmega     = SOIS.shape[1]
                minList_sq = np.full(nOmega, np.inf, dtype=float)
                SOIS_norm2 = np.sum(SOIS * SOIS, axis=0)
                for j0 in range(0, X.shape[1], block):
                    Xb         = X[:, j0:j0 + block]
                    Xb_norm2   = np.sum(Xb * Xb, axis=0)
                    C          = SOIS_norm2[:, None] + Xb_norm2[None, :] - 2.0 * (SOIS.T @ Xb)
                    minList_sq = np.minimum(minList_sq, np.min(C, axis=1))
                idx = int(np.argmax(minList_sq))
                return (float(minList_sq[idx]), idx, minList_sq)
        norms2 = np.sum(SOIS * SOIS, axis=0)
        idx0   = int(np.argmax(norms2))
        if BVP:
            state0, costate0, vf0, _ = model.solveMPCBVP(SOIS[:, idx0], T, deltaT, numbOfEval)
        else:
            state0, costate0, vf0, _ = model.solveMPCIterativ(SOIS[:, idx0], T, deltaT, numbOfEval)
        state0       = np.asarray(state0)
        costate0     = np.asarray(costate0)
        vf0          = np.asarray(vf0)
        self.state   = state0
        self.costate = costate0
        self.vf      = vf0
        if saveName is not None:
            self.save(saveName)
        currentFD, idx, minList = _fill_distance_sq(SOIS, self.state)
        if self.state.shape[1] == 0:
            raise RuntimeError('No state points remain after the cube cut. Check whether cubeBound matches the trajectories or whether the cut is too restrictive.')
        iter_count = 1
        print(f'{iter_count} iteration steps with an fill distance of {np.sqrt(currentFD)}')
        print('------------------------------------------------------------------------------')
        iter_count = 2
        while np.sqrt(currentFD) > epsFD and iter_count <= nMaxFD:
            print(idx)
            if BVP:
                stateTemp, costateTemp, vfTemp, _ = model.solveMPCBVP(SOIS[:, idx], T, deltaT, numbOfEval)
            else:
                stateTemp, costateTemp, vfTemp, _ = model.solveMPCIterativ(SOIS[:, idx], T, deltaT, numbOfEval)
            stateTemp   = np.asarray(stateTemp)
            costateTemp = np.asarray(costateTemp)
            vfTemp      = np.asarray(vfTemp)
            if len(stateTemp.shape) == 2:
                self.state   = np.concatenate([self.state, stateTemp], axis=1)
                self.costate = np.concatenate([self.costate, costateTemp], axis=1)
                self.vf      = np.concatenate([self.vf, vfTemp], axis=1)
            else:
                self.state   = np.concatenate([self.state, np.atleast_2d(stateTemp).T], axis=1)
                self.costate = np.concatenate([self.costate, np.atleast_2d(costateTemp).T], axis=1)
                self.vf      = np.concatenate([self.vf, np.atleast_2d(vfTemp)], axis=1)
            before_n = self.state.shape[1]
            after_n  = self.state.shape[1]
            if after_n == 0:
                raise RuntimeError('No state points remain after the cube cut with the newly added trajectory. Check cubeBound and the trajectories.')
            currentFD, idx, minList = _fill_distance_sq(SOIS, self.state)
            if saveName is not None:
                self.save(saveName)
            print(f'{iter_count} iteration steps with an fill distance of {np.sqrt(currentFD)}')
            if after_n != before_n:
                print(f'(cube cut removed {before_n - after_n} points)')
            print('------------------------------------------------------------------------------')
            iter_count += 1

    def makeTrainDataRandom(self, model, T, deltaT, numbOfEval, omega, numberOFTestData, seed, saveName=None):
        """Generate random training data from the initial-state grid and save it optionally."""
        np.random.seed(seed)
        idx          = np.random.choice(omega.shape[1], size=numberOFTestData, replace=False)
        self.state   = np.empty((omega.shape[0], 0))
        self.costate = np.empty((omega.shape[0], 0))
        self.vf      = np.empty((1, 0))
        for i in range(numberOFTestData):
            stateTemp, costateTemp, vfTemp, _ = model.solveMPCBVP(omega[:, idx[i]], T, deltaT, numbOfEval)
            if len(stateTemp.shape) == 2:
                self.state   = np.c_[self.state, np.atleast_2d(stateTemp[:, 0]).T]
                self.costate = np.c_[self.costate, np.atleast_2d(costateTemp[:, 0]).T]
                self.vf      = np.c_[self.vf, np.atleast_2d(vfTemp[0, 0]).T]
            else:
                self.state   = np.c_[self.state, np.atleast_2d(stateTemp).T]
                self.costate = np.c_[self.costate, np.atleast_2d(costateTemp).T]
                self.vf      = np.c_[self.vf, np.atleast_2d(vfTemp).T]
        if saveName is not None:
            self.save(saveName)

    def cutDataToCube(self, cubeBound, saveName=None):
        """Filter states, costates, and values to the prescribed box and save them optionally."""
        NCub = cubeBound.shape[0]
        for i in range(NCub):
            idx          = (self.state[i, :] >= cubeBound[i, 0]) & (self.state[i, :] <= cubeBound[i, 1])
            self.state   = self.state[:, idx]
            self.costate = self.costate[:, idx]
            self.vf      = self.vf[:, idx]
        if saveName is not None:
            self.save(saveName)


class TestData(Data):
    """Dataset class for test states, costates, and value-function data."""

    def __init__(self, name=None):
        """Initialize the object and store the required parameters."""
        super().__init__(name)

    def load(self, name):
        """Load stored data or a stored surrogate from the data folder."""
        with open('data/' + name, 'rb') as f:
            self.state, self.costate, self.vf = pickle.load(f)

    def save(self, name):
        """Save the current data object in the data folder."""
        with open('data/' + name, 'wb') as f:
            pickle.dump((self.state, self.costate, self.vf), f, protocol=4)

    def makeTestData(self, model, T, deltaT, numbOfEval, omega, numberOFTestData, seed, saveName=None, BVP=True):
        """Generate random test data from the initial-state grid and save it optionally."""
        np.random.seed(seed)
        idx          = np.random.choice(omega.shape[1], size=numberOFTestData, replace=False)
        self.state   = np.empty((omega.shape[0], 0))
        self.costate = np.empty((omega.shape[0], 0))
        self.vf      = np.empty((1, 0))
        for i in range(numberOFTestData):
            if BVP:
                stateTemp, costateTemp, vfTemp, _ = model.solveMPCBVP(omega[:, idx[i]], T, deltaT, numbOfEval)
            else:
                stateTemp, costateTemp, vfTemp, _ = model.solveMPCIterativ(omega[:, idx[i]], T, deltaT, numbOfEval)
            if len(stateTemp.shape) == 2:
                self.state   = np.c_[self.state, np.atleast_2d(stateTemp[:, 0]).T]
                self.costate = np.c_[self.costate, np.atleast_2d(costateTemp[:, 0]).T]
                self.vf      = np.c_[self.vf, np.atleast_2d(vfTemp[0, 0]).T]
            else:
                self.state   = np.c_[self.state, np.atleast_2d(stateTemp).T]
                self.costate = np.c_[self.costate, np.atleast_2d(costateTemp).T]
                self.vf      = np.c_[self.vf, np.atleast_2d(vfTemp).T]
        if saveName is not None:
            self.save(saveName)
