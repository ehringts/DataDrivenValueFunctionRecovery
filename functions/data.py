import numpy                as np
from   scipy.integrate      import solve_bvp
import matplotlib.pyplot    as plt
import time
from   scipy.interpolate    import interp1d
from   scipy.integrate      import solve_ivp, quad
import matplotlib.animation as animation
import abc
from scipy                  import linalg as la
import pickle
import random
# -----------------------------------------------------------------------------
# This class provides routines to
#   (i)  implement optimal control problems of the form int_0^\infty h(x) + u^\top R u dt subject to x'=f(x)+g(x)u, x(0)=x0
#   (ii) solve the corresponding open-loop boundary value problem
# -----------------------------------------------------------------------------


class Data(metaclass=abc.ABCMeta):
    def __init__(self, name = None):
        self.name = name
        if name is not None:
           self.load(name)

    @abc.abstractmethod
    def load(self): 
        pass    
    @abc.abstractmethod
    def save(self): 
        pass      
  
class TrainData(Data): # Van der Pol oscillator
    def __init__(self, name = None):
        super().__init__(name)

    def load(self,name):
        with open("data/"+name, 'rb') as f:
             self.state,self.costate,self.vf = pickle.load(f)

    def save(self,name):
        with open("data/"+name, 'wb') as f:
             pickle.dump((self.state,self.costate,self.vf),f, protocol=4)             


    def makeTrainData(self, model, T, deltaT, numbOfEval, omega, epsFD, nMaxFD, cubeBound, saveName=None, BVP = True):
        """
        Greedy Datensatz-Generierung über Fill-Distance:
        - Wähle jeweils den omega-Punkt, der maximal weit vom aktuellen Sample-Set (self.state) entfernt ist.
        - Löse BVP, füge Trajektorie hinzu.
        Verbesserungen:
        (1) schneller: nearest-neighbor via cKDTree (SciPy), Fallback: chunked NumPy
        (2) korrekt bei Löschungen: nach jedem Cube-Cut wird minList neu berechnet.
        
        Konventionen:
        - currentFD und minList sind quadratische Distanzen (wie in deinem Original).
        - cubeBound schneidet nur die ersten NCub State-Komponenten (wie cutDataToCube).
        """
        import numpy as np

        SOIS = np.asarray(omega, dtype=float)
        cubeBound = np.asarray(cubeBound, dtype=float)

        if SOIS.ndim != 2:
            raise ValueError(f"omega muss 2D sein (d, nOmega). Bekommen: {SOIS.shape}")
        if cubeBound.ndim != 2 or cubeBound.shape[1] != 2:
            raise ValueError(f"cubeBound muss Shape (NCub, 2) haben. Bekommen: {cubeBound.shape}")

        d, nOmega = SOIS.shape
        NCub = cubeBound.shape[0]

        # --- Helper: Filter Trajektorie direkt in den Cube (spart Speicher + verhindert spätere Löschungen)
        def _mask_in_cube(X):
            # X: (d, m)
            if X.size == 0:
                return np.zeros((0,), dtype=bool)
            m = X.shape[1]
            mask = np.ones(m, dtype=bool)
            for i in range(min(NCub, X.shape[0])):  # nur die Komponenten, die cubeBound definiert
                mask &= (X[i, :] >= cubeBound[i, 0]) & (X[i, :] <= cubeBound[i, 1])
            return mask

        def _apply_cube_filter(state, costate, vf):
            mask = _mask_in_cube(state)
            return state[:, mask], costate[:, mask], vf[:, mask]

        # --- Helper: Fill distance (quadratisch) + minList (quadratisch)
        # Prefer SciPy KDTree; fallback to chunked NumPy if SciPy not installed.
        try:
            from scipy.spatial import cKDTree

            def _fill_distance_sq(SOIS, X):
                # returns: (value_sq, idx, minList_sq)
                if X is None or X.shape[1] == 0:
                    # kein Sample -> wähle omega mit größter Norm (wie dein Start)
                    norms2 = np.sum(SOIS * SOIS, axis=0)
                    idx = int(np.argmax(norms2))
                    minList_sq = np.full(SOIS.shape[1], np.inf, dtype=float)
                    return float(np.inf), idx, minList_sq

                tree = cKDTree(X.T)
                dist = tree.query(SOIS.T, k=1, workers=-1)[0]  # echte Distanz
                minList_sq = dist * dist
                idx = int(np.argmax(minList_sq))
                return float(minList_sq[idx]), idx, minList_sq

        except Exception:
            # Fallback ohne SciPy (chunked, damit keine gigantische Matrix entsteht)
            def _fill_distance_sq(SOIS, X, block=2048):
                if X is None or X.shape[1] == 0:
                    norms2 = np.sum(SOIS * SOIS, axis=0)
                    idx = int(np.argmax(norms2))
                    minList_sq = np.full(SOIS.shape[1], np.inf, dtype=float)
                    return float(np.inf), idx, minList_sq

                nOmega = SOIS.shape[1]
                minList_sq = np.full(nOmega, np.inf, dtype=float)

                SOIS_norm2 = np.sum(SOIS * SOIS, axis=0)  # (nOmega,)

                for j0 in range(0, X.shape[1], block):
                    Xb = X[:, j0:j0 + block]               # (d, b)
                    Xb_norm2 = np.sum(Xb * Xb, axis=0)     # (b,)
                    # C = ||s - x||^2 = s^2 + x^2 - 2 s^T x
                    C = (SOIS_norm2[:, None] + Xb_norm2[None, :] - 2.0 * (SOIS.T @ Xb))
                    minList_sq = np.minimum(minList_sq, np.min(C, axis=1))

                idx = int(np.argmax(minList_sq))
                return float(minList_sq[idx]), idx, minList_sq

        # --- Startindex: wie vorher (max Norm)
        norms2 = np.sum(SOIS * SOIS, axis=0)
        idx0 = int(np.argmax(norms2))

        # --- Initial solve
        if BVP:
            state0, costate0, vf0, _ = model.solveMPCBVP(SOIS[:, idx0], T, deltaT, numbOfEval)
        else:
            state0, costate0, vf0, _ = model.solveMPCIterativ(SOIS[:, idx0], T, deltaT, numbOfEval)

        state0 = np.asarray(state0)
        costate0 = np.asarray(costate0)
        vf0 = np.asarray(vf0)

        # Direkt auf Cube filtern (statt später groß zu löschen)
        #state0, costate0, vf0 = _apply_cube_filter(state0, costate0, vf0)

        self.state = state0
        self.costate = costate0
        self.vf = vf0

        if saveName is not None:
            self.save(saveName)

        # --- minList & FD initial (nach Cube)
        currentFD, idx, minList = _fill_distance_sq(SOIS, self.state)

        # Wenn durch Cube alles weg ist, lieber sauber abbrechen statt Endlosschleife
        if self.state.shape[1] == 0:
            raise RuntimeError(
                "Nach dem Cube-Cut sind keine State-Punkte übrig geblieben. "
                "Entweder cubeBound passt nicht zu den Trajektorien oder du schneidest zu aggressiv."
            )

        iter_count = 1
        print(f"{iter_count} iteration steps with an fill distance of {np.sqrt(currentFD) }")
        print("------------------------------------------------------------------------------")

        iter_count = 2
        while np.sqrt(currentFD) > epsFD and iter_count <= nMaxFD:
            # idx ist der nächste omega-Punkt, den wir sampeln
            print(idx)
            if BVP:
                stateTemp, costateTemp, vfTemp, _ = model.solveMPCBVP(SOIS[:, idx], T, deltaT, numbOfEval)
            else:
                stateTemp, costateTemp, vfTemp, _ = model.solveMPCIterativ(SOIS[:, idx], T, deltaT, numbOfEval)

            stateTemp = np.asarray(stateTemp)
            costateTemp = np.asarray(costateTemp)
            vfTemp = np.asarray(vfTemp)

            # Direkt in Cube filtern (spart concat+delete)
            #stateTemp, costateTemp, vfTemp = _apply_cube_filter(stateTemp, costateTemp, vfTemp)

            # Falls neue Trajektorie komplett rausfliegt, überspringen wir das Update
            #if stateTemp.shape[1] > 0:
                # Append (ja, concat kopiert; aber durch Cube-Filter oft deutlich kleiner)
            if len(stateTemp.shape)==2:    
                self.state = np.concatenate([self.state, stateTemp], axis=1)
                self.costate = np.concatenate([self.costate, costateTemp], axis=1)
                self.vf = np.concatenate([self.vf, vfTemp], axis=1)
            else:    
                self.state = np.concatenate([self.state, np.atleast_2d(stateTemp).T], axis=1)
                self.costate = np.concatenate([self.costate, np.atleast_2d(costateTemp).T], axis=1)
                self.vf = np.concatenate([self.vf, np.atleast_2d(vfTemp)], axis=1)
            # Sicherheit: nochmal cutten (falls du oben nicht alles abdeckst / numerische Randfälle)
            before_n = self.state.shape[1]
            #self.cutDataToCube(cubeBound)
            after_n = self.state.shape[1]

            if after_n == 0:
                raise RuntimeError(
                    "Nach dem Cube-Cut sind keine State-Punkte übrig geblieben (mit den neu hinzugefügten). "
                    "cubeBound/Trajektorien prüfen."
                )

            # --- KORREKTUR bei Löschungen:
            # Egal ob gelöscht wurde oder nicht: wir berechnen minList neu (immer konsistent).
            # (KDTree ist schnell genug und verhindert minList-Inkonsistenzen durch deletions.)
            currentFD, idx, minList = _fill_distance_sq(SOIS, self.state)

            if saveName is not None:
                self.save(saveName)

            print(f"{iter_count} iteration steps with an fill distance of {np.sqrt(currentFD) }")
            if after_n != before_n:
                print(f"(cube cut removed {before_n - after_n} points)")
            print("------------------------------------------------------------------------------")

            iter_count += 1

    def makeTrainDataRandom(self,model,T,deltaT,numbOfEval,omega,numberOFTestData,seed,saveName = None):
        np.random.seed(seed)  
        idx =  np.random.choice(omega.shape[1], size=numberOFTestData, replace=False)


        self.state = np.empty((omega.shape[0],0))
        self.costate = np.empty((omega.shape[0],0))
        self.vf = np.empty((1,0))

        for i in range(numberOFTestData):   
            stateTemp,costateTemp,vfTemp,_ = model.solveMPCBVP(omega[:,idx[i]],T,deltaT,numbOfEval)
            if len(stateTemp.shape) == 2:
                self.state                     = np.c_[self.state,np.atleast_2d(stateTemp[:,0]).T]
                self.costate                   = np.c_[self.costate,np.atleast_2d(costateTemp[:,0]).T ]
                self.vf                        = np.c_[self.vf,np.atleast_2d(vfTemp[0,0]).T]
            else:  
                self.state                     = np.c_[self.state,np.atleast_2d(stateTemp).T]
                self.costate                   = np.c_[self.costate,np.atleast_2d(costateTemp).T ]
                self.vf                        = np.c_[self.vf,np.atleast_2d(vfTemp).T]  

        if saveName is not None:
            self.save(saveName)

    def cutDataToCube(self,cubeBound,saveName=None):
        NCub = cubeBound.shape[0]
        for i in range(NCub):
            idx          = (self.state[i,:]>=cubeBound[i,0]) & (self.state[i,:]<=cubeBound[i,1])
            self.state   = self.state[:,idx]
            self.costate = self.costate[:,idx]
            self.vf      = self.vf[:,idx]
        if saveName is not None:
                 self.save(saveName)
    

class TestData(Data): # Van der Pol oscillator
    def __init__(self, name = None):
        super().__init__(name)

    def load(self,name):
        with open("data/"+name, 'rb') as f:
             self.state,self.costate,self.vf = pickle.load(f)

    def save(self,name):
        with open("data/"+name, 'wb') as f:
             pickle.dump((self.state,self.costate,self.vf),f, protocol=4)             


    def makeTestData(self,model,T,deltaT,numbOfEval,omega,numberOFTestData,seed,saveName = None,BVP = True):
        np.random.seed(seed)  
        idx =  np.random.choice(omega.shape[1], size=numberOFTestData, replace=False)


        self.state = np.empty((omega.shape[0],0))
        self.costate = np.empty((omega.shape[0],0))
        self.vf = np.empty((1,0))

        for i in range(numberOFTestData):   
            if BVP:
                stateTemp,costateTemp,vfTemp,_  = model.solveMPCBVP(omega[:,idx[i]],T,deltaT,numbOfEval)
            else:
                stateTemp,costateTemp,vfTemp,_  = model.solveMPCIterativ(omega[:,idx[i]],T,deltaT,numbOfEval)
            if len(stateTemp.shape) == 2:
                self.state                     = np.c_[self.state,np.atleast_2d(stateTemp[:,0]).T]
                self.costate                   = np.c_[self.costate,np.atleast_2d(costateTemp[:,0]).T ]
                self.vf                        = np.c_[self.vf,np.atleast_2d(vfTemp[0,0]).T]
            else:  
                self.state                     = np.c_[self.state,np.atleast_2d(stateTemp).T]
                self.costate                   = np.c_[self.costate,np.atleast_2d(costateTemp).T ]
                self.vf                        = np.c_[self.vf,np.atleast_2d(vfTemp).T]  

        if saveName is not None:
            self.save(saveName)

