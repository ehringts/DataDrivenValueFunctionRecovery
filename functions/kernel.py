import numpy as np
import abc

# -----------------------------------------------------------------------------
# This class provides routines to
#   (i)  compute the generalized Gramian matrix used in the RKHS-PI method, and
#   (ii) evaluate the surrogate model and its gradient.
# Multiple kernel choices are supported.
# -----------------------------------------------------------------------------

class Kernel(metaclass=abc.ABCMeta):
    """
    Abstract base for radial kernels used in RKHS-PI constructions.
    kernels have the form k(x,y) = phi(||x-y||) 

    Subclasses must implement:
        - phi(r)   : radial basis function
        - phiR(r)  : phiR(r)  = d/dr phi(r)  * (1/r)
        - phiRR(r) : phiRR(r) = d/dr phiR(r) * (1/r)

    Parameters
    ----------
    gamma : float
        Kernel shape parameter.
    """
    def __init__(self,gamma):
        self.gamma    = gamma
    
    def setGamma(self,gamma):
        self.gamma = gamma

    # --- Abstract radial functions ----------------------------------------
    @abc.abstractmethod
    def phi(self,r): 
        pass    
    @abc.abstractmethod
    def phiR(self,r): 
        pass    
    @abc.abstractmethod
    def phiRR(self,r): 
        pass         

    def getGramHermite(self, y, funcYList, x, funcXList):
        y = np.asarray(y)
        x = np.asarray(x)
        fy = np.asarray(funcYList, dtype=int)  # (ny,)
        fx = np.asarray(funcXList, dtype=int)  # (nx,)

        ny, nx = y.shape[1], x.shape[1]

        # Pairwise distances between columns of y and x: (ny, nx)
        x2 = np.sum(x * x, axis=0)                 # (nx,)
        y2 = np.sum(y * y, axis=0)                 # (ny,)
        diff2 = y2[:, None] + x2[None, :] - 2.0 * (y.T @ x)
        diff = np.sqrt(np.abs(diff2))

        K = np.zeros((ny, nx), dtype=diff.dtype)

        iy0 = (fy == 0)
        ix0 = (fx == 0)
        iyp = ~iy0
        ixp = ~ix0

        # --- Case (0,0): phi(diff)
        if np.any(iy0) and np.any(ix0):
            idx = np.ix_(iy0, ix0)
            K[idx] = self.phi(diff[idx])

        # --- Precompute for columns with fx>0 (needed in multiple cases)
        if np.any(ixp):
            cols_p = np.flatnonzero(ixp)           # indices of x-columns with fx>0
            fx_idx = fx[ixp] - 1                  # 0-based dim indices, (nxp,)

            x_fx = x[fx_idx, cols_p]              # (nxp,)  x[fx[j]-1, j]
            y_fx_T = y[fx_idx, :].T               # (ny, nxp) y[fx[j]-1, i]
            B_full = x_fx[None, :] - y_fx_T       # (ny, nxp)  (x_dim - y_dim) for fx

        # --- Precompute for rows with fy>0 (needed in multiple cases)
        if np.any(iyp):
            rows_p = np.flatnonzero(iyp)          # indices of y-columns with fy>0
            fy_idx = fy[iyp] - 1                  # 0-based dim indices, (nyp,)

            y_fy = y[fy_idx, rows_p]              # (nyp,)  y[fy[i]-1, i]
            x_fy = x[fy_idx, :]                   # (nyp, nx) x[fy[i]-1, j]
            A_full = y_fy[:, None] - x_fy         # (nyp, nx) (y_dim - x_dim) for fy

        # --- Case (0, >0): phiR(diff) * (x_dim - y_dim)
        if np.any(iy0) and np.any(ixp):
            idx = np.ix_(iy0, ixp)
            K[idx] = self.phiR(diff[idx]) * B_full[iy0, :]

        # --- Case (>0, 0): phiR(diff) * (y_dim - x_dim)
        if np.any(iyp) and np.any(ix0):
            idx = np.ix_(iyp, ix0)
            K[idx] = self.phiR(diff[idx]) * A_full[:, ix0]

        # --- Case (>0, >0):
        # phiRR(diff) * (y_fy - x_fy) * (x_fx - y_fx)  minus phiR(diff) if fy==fx
        if np.any(iyp) and np.any(ixp):
            rows_p = np.flatnonzero(iyp)
            cols_p = np.flatnonzero(ixp)

            idx = np.ix_(iyp, ixp)

            A = A_full[:, ixp]                    # (nyp, nxp)
            B = B_full[rows_p, :]                 # (nyp, nxp)

            K[idx] = self.phiRR(diff[idx]) * (A * B)

            eq = (fy[iyp][:, None] == fx[ixp][None, :])  # (nyp, nxp)
            K[idx] -= self.phiR(diff[idx]) * eq

        return K

    def getGramControl(self, y, funcYList, x, funcXList, R, g):
        """
        Fast Gram matrix for control functionals for

            k(y,x) = phi(||y-x||)

        funcYList and funcXList encode control components 1,...,m.
        """

        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        d, Ny = y.shape
        d2, Nx = x.shape

        if d != d2:
            raise ValueError("y and x must have the same state dimension.")

        funcY = np.asarray(funcYList, dtype=int).reshape(-1)
        funcX = np.asarray(funcXList, dtype=int).reshape(-1)

        def batch_g(Z):
            Z = np.asarray(Z, dtype=float)
            dz, N = Z.shape

            G = np.asarray(g(Z), dtype=float)

            if G.ndim == 1:
                return G.reshape(dz, 1, 1)

            if G.ndim == 2:
                if G.shape[0] != dz:
                    raise ValueError("g(Z) returned 2D array with wrong first dimension.")

                if G.shape[1] == N and N > 1:
                    return G[:, None, :]

                m = G.shape[1]
                return np.broadcast_to(G[:, :, None], (dz, m, N))

            if G.ndim == 3:
                if G.shape[0] == dz and G.shape[2] == N:
                    return G
                if G.shape[0] == N and G.shape[1] == dz:
                    return np.transpose(G, (1, 2, 0))
                if G.shape[0] == dz and G.shape[1] == N:
                    return np.transpose(G, (0, 2, 1))

            raise ValueError("g(Z) has unsupported shape.")

        GY = batch_g(y)
        GX = batch_g(x)

        m = GY.shape[1]

        R = np.asarray(R, dtype=float)

        if R.ndim == 0:
            Rmat = float(R) * np.eye(m)
        elif R.ndim == 1:
            Rmat = np.diag(R)
        else:
            Rmat = R

        BY = GY.transpose(1, 0, 2).reshape(m, d * Ny)
        BX = GX.transpose(1, 0, 2).reshape(m, d * Nx)

        AallY = -0.5 * np.linalg.solve(Rmat, BY)
        AallX = -0.5 * np.linalg.solve(Rmat, BX)

        AallY = AallY.reshape(m, d, Ny).transpose(2, 0, 1)
        AallX = AallX.reshape(m, d, Nx).transpose(2, 0, 1)

        compY = funcY - 1
        compX = funcX - 1

        AY = AallY[np.arange(Ny), compY, :]
        AX = AallX[np.arange(Nx), compX, :]

        yT = y.T
        xT = x.T

        yy = np.sum(y * y, axis=0)[:, None]
        xx = np.sum(x * x, axis=0)[None, :]

        sq = yy + xx - 2.0 * (yT @ x)
        sq = np.maximum(sq, 0.0)

        r = np.sqrt(sq)

        phiR  = self.phiR(r)
        phiRR = self.phiRR(r)

        AY_x = AY @ x
        AY_y = np.sum(AY * yT, axis=1)[:, None]
        AY_h = AY_x - AY_y

        AX_x = np.sum(AX * xT, axis=1)[None, :]
        AX_y = yT @ AX.T
        AX_h = AX_x - AX_y

        AY_AX = AY @ AX.T

        K = -phiRR * AY_h * AX_h - phiR * AY_AX

        return K

# ---------------------------------------------------------------------------
# Concrete kernels
# ---------------------------------------------------------------------------

class QuadWendland(Kernel):
    def __init__(self,gamma,d):
        self.l        = np.floor(d/2)+ 2 +1 
        self.gamma    = gamma
    def phi(self,r):   return (self.gamma*r<=1) * (1-self.gamma*r)**(self.l+2) * ((self.l**2+4*self.l+3)*(self.gamma*r)**2+(3*self.l+6)*self.gamma*r+3)  
    def phiR(self,r):  return (self.gamma*r<=1) * (1-self.gamma*r)**(self.l+1) * (-self.gamma**2) * (12+7*self.l+self.l**2)*(1+(1+self.l)*self.gamma*r)
    def phiRR(self,r): return (self.gamma*r<=1) * (1-self.gamma*r)**(self.l) * self.gamma**4 * (24 + 50 * self.l + 35 *  self.l**2 + 10 *  self.l**3 + self.l**4) 

class QuadMatern(Kernel):
    def phi(self,r):   return np.exp(-self.gamma*r)*(3+3*self.gamma*r+self.gamma**2 *r**2)   
    def phiR(self,r):  return (-1)*np.exp(-self.gamma*r)*(1+self.gamma*r) * self.gamma**2 
    def phiRR(self,r): return self.gamma**4 * np.exp(-self.gamma*r) 

class Gauss(Kernel):
    def phi(self,r):   return np.exp(-self.gamma*(r**2)) 
    def phiR(self,r):  return (-2)*self.gamma*np.exp(-self.gamma*(r**2)) 
    def phiRR(self,r): return 4*self.gamma**2 * np.exp(-self.gamma*(r**2)) 

class InvMulti(Kernel):
    def phi(self,r):   return 1/np.sqrt(1+self.gamma*(r**2)) 
    def phiR(self,r):  return (-1)*self.gamma* (1/np.sqrt((1+self.gamma*(r**2))**3)) 
    def phiRR(self,r): return 3*self.gamma**2 * (1/np.sqrt((1+self.gamma*(r**2))**5)) 

class LinMatern(Kernel):
    def phi(self,r):   return np.exp(-self.gamma*r)*(1+self.gamma*r)   
    def phiR(self,r):  return (-1)*np.exp(-self.gamma*r)*self.gamma**2 
    def phiRR(self,r): 
        diffMask1 = r<10**(-14)
        diffMask2 = r>10**(-14)
        return self.gamma**3 * np.exp(-self.gamma*r) * (1/(r+diffMask1)) * (diffMask2)

class CubMatern(Kernel):  # nu = 7/2
    def phi(self, r):
        g = self.gamma
        return np.exp(-g*r) * (15 + 15*g*r + 6*(g**2)*(r**2) + (g**3)*(r**3))

    def phiR(self, r):
        g = self.gamma
        return (-1) * np.exp(-g*r) * (3 + 3*g*r + (g**2)*(r**2)) * (g**2)

    def phiRR(self, r):
        g = self.gamma
        return (g**4) * np.exp(-g*r) * (1 + g*r)


class QuartMatern(Kernel):  # nu = 9/2
    def phi(self, r):
        g = self.gamma
        return np.exp(-g*r) * (
            105 + 105*g*r + 45*(g**2)*(r**2) + 10*(g**3)*(r**3) + (g**4)*(r**4)
        )

    def phiR(self, r):
        g = self.gamma
        return (-1) * np.exp(-g*r) * (15 + 15*g*r + 6*(g**2)*(r**2) + (g**3)*(r**3)) * (g**2)

    def phiRR(self, r):
        g = self.gamma
        return (g**4) * np.exp(-g*r) * (3 + 3*g*r + (g**2)*(r**2)) 


class KernelProduct(metaclass=abc.ABCMeta):
    """
    Like Kernel, but with an additional linear factor (y^T x)^2
    woven into the expressions; kernels have the form k(x,y) = phi(||x-y||) * (y^T x)^2;
    'case' controls which degree is used.

    Parameters
    ----------
    gamma : float

    """
    def __init__(self,gamma):
        self.gamma    = gamma
    
    def setGamma(self,gamma):
        self.gamma = gamma

    # --- Abstract radial functions ----------------------------------------
    @abc.abstractmethod
    def phi(self,r): 
        pass    
    @abc.abstractmethod
    def phiR(self,r): 
        pass    
    @abc.abstractmethod
    def phiRR(self,r): 
        pass         

   
    def getGramHermite(self, y, funcYList, x, funcXList):
        """
        Vectorized Hermite Gram for k(x,y) = phi(||x-y||) * (y^T x)^2
        where func lists encode:
        0 -> value
        p>0 -> partial derivative wrt coordinate p (1-based)
        x, y are shaped (d, Nx/Ny)
        """
        y = np.asarray(y)
        x = np.asarray(x)
        funcY = np.asarray(funcYList, dtype=np.int64)
        funcX = np.asarray(funcXList, dtype=np.int64)

        d, Ny = y.shape
        d2, Nx = x.shape
        if d != d2:
            raise ValueError("x and y must have the same first dimension (ambient dimension).")
        if funcY.shape[0] != Ny or funcX.shape[0] != Nx:
            raise ValueError("funcYList length must be Ny and funcXList length must be Nx.")
        if np.any(funcY < 0) or np.any(funcY > d) or np.any(funcX < 0) or np.any(funcX > d):
            raise ValueError("Functional indices must be in {0,1,...,d} (1-based for derivatives).")

        # r_ij = ||x_j - y_i||
        xx = np.sum(x * x, axis=0, keepdims=True)          # (1, Nx)
        yy = np.sum(y * y, axis=0, keepdims=True).T        # (Ny, 1)
        sq = yy + xx - 2.0 * (y.T @ x)                     # (Ny, Nx)
        sq = np.maximum(sq, 0.0)                           # numerical safety
        r = np.sqrt(sq)

        # s_ij = y_i^T x_j
        s = y.T @ x                                        # (Ny, Nx)

        # Kernel radial pieces (assumed to be numpy-vectorized)
        phi   = self.phi(r)
        phiR  = self.phiR(r)
        phiRR = self.phiRR(r)

        # Build functional grids
        fy = funcY[:, None]        # (Ny, 1)
        fx = funcX[None, :]        # (1, Nx)

        m00 = (fy == 0) & (fx == 0)
        m0x = (fy == 0) & (fx > 0)
        my0 = (fy > 0) & (fx == 0)
        myx = (fy > 0) & (fx > 0)

        # p depends on columns (x functionals), q depends on rows (y functionals)
        p_cols_safe = np.where(funcX > 0, funcX - 1, 0)    # (Nx,)
        q_rows_safe = np.where(funcY > 0, funcY - 1, 0)    # (Ny,)

        j = np.arange(Nx)
        i = np.arange(Ny)

        # x_p(j) and y_p(i,j) where p=p_j
        x_p = x[p_cols_safe, j]                            # (Nx,)
        y_p = y[p_cols_safe, :].T                          # (Ny, Nx)  entry (i,j)=y[p_j,i]

        # y_q(i) and x_q(i,j) where q=q_i
        y_q = y[q_rows_safe, i]                            # (Ny,)
        x_q = x[q_rows_safe, :]                            # (Ny, Nx)  entry (i,j)=x[q_i,j]

        # d_p = x_p - y_p  and d_q = x_q - y_q
        dp = x_p[None, :] - y_p                            # (Ny, Nx)  entry (i,j)=x[p_j,j]-y[p_j,i]
        dq = x_q - y_q[:, None]                            # (Ny, Nx)  entry (i,j)=x[q_i,j]-y[q_i,i]

        # delta_{pq}: p_j == q_i
        delta = (q_rows_safe[:, None] == p_cols_safe[None, :])  # (Ny, Nx) boolean

        # Assemble K
        K = np.zeros((Ny, Nx), dtype=np.result_type(phi, s, r))

        s2 = s * s

        # value-value
        K[m00] = (phi * s2)[m00]

        # value - x-derivative (∂/∂x_p)
        #   = s^2 * phiR * d_p + 2 * phi * s * y_p
        tmp0x = s2 * phiR * dp + 2.0 * phi * s * y_p
        K[m0x] = tmp0x[m0x]

        # y-derivative - value (∂/∂y_q)
        #   = - s^2 * phiR * d_q + 2 * phi * s * x_q
        tmpy0 = -s2 * phiR * dq + 2.0 * phi * s * x_q
        K[my0] = tmpy0[my0]

        # y-derivative - x-derivative (∂^2 / ∂y_q ∂x_p)
        #   = -s^2*phiRR*d_p*d_q - s^2*phiR*delta
        #     + 2*s*phiR*(d_p*x_q - d_q*y_p)
        #     + 2*phi*x_q*y_p + 2*phi*s*delta
        tmpyx = (
            -s2 * phiRR * dp * dq
            -s2 * phiR * delta
            + 2.0 * s * phiR * (dp * x_q - dq * y_p)
            + 2.0 * phi * x_q * y_p
            + 2.0 * phi * s * delta
        )
        K[myx] = tmpyx[myx]

        return K    
    

    def getGramControl(self, y, funcYList, x, funcXList, R, g):
        """
        Fast Gram matrix for control functionals for

            k(y,x) = phi(||y-x||) * (y^T x)^2

        Control functional:

            L_{z,i} V = (-1/2 * R^{-1} g(z)^T grad V(z))_i

        funcYList and funcXList encode control components 1,...,m.
        """

        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        d, Ny = y.shape
        d2, Nx = x.shape

        if d != d2:
            raise ValueError("y and x must have the same state dimension.")

        funcY = np.asarray(funcYList, dtype=int).reshape(-1)
        funcX = np.asarray(funcXList, dtype=int).reshape(-1)

        if funcY.size != Ny:
            raise ValueError("funcYList must have length y.shape[1].")
        if funcX.size != Nx:
            raise ValueError("funcXList must have length x.shape[1].")

        # ------------------------------------------------------------
        # Normalize batched g output to shape (d, m, N)
        # ------------------------------------------------------------

        def batch_g(Z):
            """
            Required/accepted outputs of g(Z):

                (d,m)       constant control matrix
                (d,m,N)     batched control matrix
                (N,d,m)     batched control matrix
                (d,N,m)     batched control matrix
                (d,N)       scalar-control batched matrix, interpreted as m=1

            Returns:
                G shape (d,m,N)
            """
            Z = np.asarray(Z, dtype=float)
            dz, N = Z.shape

            G = np.asarray(g(Z), dtype=float)

            if G.ndim == 1:
                if G.size != dz:
                    raise ValueError("g(Z) returned vector with wrong length.")
                return G.reshape(dz, 1, 1)

            if G.ndim == 2:
                if G.shape[0] != dz:
                    raise ValueError("g(Z) returned 2D array with wrong first dimension.")

                # scalar-control batch case: shape (d,N)
                # Ambiguous if constant m == N. In that case return 3D from g.
                if G.shape[1] == N and N > 1:
                    return G[:, None, :]          # (d,1,N)

                # constant matrix case: shape (d,m)
                m = G.shape[1]
                return np.broadcast_to(G[:, :, None], (dz, m, N))

            if G.ndim == 3:
                # already (d,m,N)
                if G.shape[0] == dz and G.shape[2] == N:
                    return G

                # (N,d,m) -> (d,m,N)
                if G.shape[0] == N and G.shape[1] == dz:
                    return np.transpose(G, (1, 2, 0))

                # (d,N,m) -> (d,m,N)
                if G.shape[0] == dz and G.shape[1] == N:
                    return np.transpose(G, (0, 2, 1))

            raise ValueError(
                "g(Z) must return shape (d,m), (d,m,N), (N,d,m), (d,N,m), or (d,N)."
            )

        GY = batch_g(y)     # (d,m,Ny)
        GX = batch_g(x)     # (d,m,Nx)

        m = GY.shape[1]

        if GX.shape[1] != m:
            raise ValueError("g(y) and g(x) have different control dimensions.")

        # ------------------------------------------------------------
        # Build R matrix
        # ------------------------------------------------------------

        R = np.asarray(R, dtype=float)

        if R.ndim == 0:
            Rmat = float(R) * np.eye(m)
        elif R.ndim == 1:
            if R.size != m:
                raise ValueError("R vector has wrong length.")
            Rmat = np.diag(R)
        elif R.ndim == 2:
            if R.shape != (m, m):
                raise ValueError("R matrix has wrong shape.")
            Rmat = R
        else:
            raise ValueError("R must be scalar, vector, or matrix.")

        # ------------------------------------------------------------
        # Compute all control coefficient matrices vectorized
        #
        # A(z) = -1/2 R^{-1} g(z)^T
        #
        # AallY shape: (Ny,m,d)
        # AallX shape: (Nx,m,d)
        # ------------------------------------------------------------

        BY = GY.transpose(1, 0, 2).reshape(m, d * Ny)   # (m, d*Ny)
        BX = GX.transpose(1, 0, 2).reshape(m, d * Nx)   # (m, d*Nx)

        AallY = -0.5 * np.linalg.solve(Rmat, BY)
        AallX = -0.5 * np.linalg.solve(Rmat, BX)

        AallY = AallY.reshape(m, d, Ny).transpose(2, 0, 1)  # (Ny,m,d)
        AallX = AallX.reshape(m, d, Nx).transpose(2, 0, 1)  # (Nx,m,d)

        compY = funcY - 1
        compX = funcX - 1

        if np.any(compY < 0) or np.any(compY >= m):
            raise ValueError("funcYList contains invalid control components.")
        if np.any(compX < 0) or np.any(compX >= m):
            raise ValueError("funcXList contains invalid control components.")

        AY = AallY[np.arange(Ny), compY, :]     # (Ny,d)
        AX = AallX[np.arange(Nx), compX, :]     # (Nx,d)

        # ------------------------------------------------------------
        # Pairwise geometry
        # ------------------------------------------------------------

        yT = y.T
        xT = x.T

        yy = np.sum(y * y, axis=0)[:, None]      # (Ny,1)
        xx = np.sum(x * x, axis=0)[None, :]      # (1,Nx)

        sq = yy + xx - 2.0 * (yT @ x)
        sq = np.maximum(sq, 0.0)

        r = np.sqrt(sq)
        s = yT @ x                               # (Ny,Nx)

        phi   = self.phi(r)
        phiR  = self.phiR(r)
        phiRR = self.phiRR(r)

        s2 = s * s

        # h = x_j - y_i
        AY_x = AY @ x                            # AY[i] @ x_j
        AY_y = np.sum(AY * yT, axis=1)[:, None]  # AY[i] @ y_i
        AY_h = AY_x - AY_y                       # AY[i] @ (x_j - y_i)

        AX_x = np.sum(AX * xT, axis=1)[None, :]  # AX[j] @ x_j
        AX_y = yT @ AX.T                         # AX[j] @ y_i
        AX_h = AX_x - AX_y                       # AX[j] @ (x_j - y_i)

        AY_AX = AY @ AX.T                        # AY[i] @ AX[j]

        # ------------------------------------------------------------
        # Contracted product-kernel mixed derivative
        # ------------------------------------------------------------

        K = (
            -s2 * phiRR * AY_h * AX_h
            -s2 * phiR  * AY_AX
            + 2.0 * s * phiR * (AX_h * AY_x - AY_h * AX_y)
            + 2.0 * phi * AY_x * AX_y
            + 2.0 * phi * s * AY_AX
        )

        return K
# ---------------------------------------------------------------------------
# Concrete product kernels
# ---------------------------------------------------------------------------

class QuadWendlandProduct(KernelProduct):
    def __init__(self,gamma,d,case = 0):
        self.l        = np.floor(d/2)+ 2 +1 
        self.case     = case
        self.gamma    = gamma
    def phi(self,r):   return (self.gamma*r<=1) * (1-self.gamma*r)**(self.l+2) * ((self.l**2+4*self.l+3)*(self.gamma*r)**2+(3*self.l+6)*self.gamma*r+3)  
    def phiR(self,r):  return (self.gamma*r<=1) * (1-self.gamma*r)**(self.l+1) * (-self.gamma**2) * (12+7*self.l+self.l**2)*(1+(1+self.l)*self.gamma*r)
    def phiRR(self,r): return (self.gamma*r<=1) * (1-self.gamma*r)**(self.l) * self.gamma**4 * (24 + 50 * self.l + 35 *  self.l**2 + 10 *  self.l**3 + self.l**4) 

class QuadMaternProduct(KernelProduct):
    def phi(self,r):   return np.exp(-self.gamma*r)*(3+3*self.gamma*r+self.gamma**2 *r**2)   
    def phiR(self,r):  return (-1)*np.exp(-self.gamma*r)*(1+self.gamma*r) * self.gamma**2 
    def phiRR(self,r): return self.gamma**4 * np.exp(-self.gamma*r) 

class GaussProduct(KernelProduct):
    def phi(self,r):   return np.exp(-self.gamma*(r**2)) 
    def phiR(self,r):  return (-2)*self.gamma*np.exp(-self.gamma*(r**2)) 
    def phiRR(self,r): return 4*self.gamma**2 * np.exp(-self.gamma*(r**2)) 

class InvMultiProduct(KernelProduct):
    def phi(self,r):   return 1/np.sqrt(1+self.gamma*(r**2)) 
    def phiR(self,r):  return (-1)*self.gamma* (1/np.sqrt((1+self.gamma*(r**2))**3)) 
    def phiRR(self,r): return 3*self.gamma**2 * (1/np.sqrt((1+self.gamma*(r**2))**5)) 

class LinMaternProduct(KernelProduct):
    def phi(self,r):   return np.exp(-self.gamma*r)*(1+self.gamma*r)   
    def phiR(self,r):  return (-1)*np.exp(-self.gamma*r)*self.gamma**2 
    def phiRR(self,r): 
        diffMask1 = r<10**(-14)
        diffMask2 = r>10**(-14)
        return self.gamma**3 * np.exp(-self.gamma*r) * (1/(r+diffMask1)) * (diffMask2)

class CubMaternProduct(KernelProduct):  # nu = 7/2
    def phi(self, r):
        g = self.gamma
        return np.exp(-g*r) * (15 + 15*g*r + 6*(g**2)*(r**2) + (g**3)*(r**3))

    def phiR(self, r):
        g = self.gamma
        return (-1) * np.exp(-g*r) * (3 + 3*g*r + (g**2)*(r**2)) * (g**2)

    def phiRR(self, r):
        g = self.gamma
        return (g**4) * np.exp(-g*r) * (1 + g*r)


class QuartMaternProduct(KernelProduct):  # nu = 9/2
    def phi(self, r):
        g = self.gamma
        return np.exp(-g*r) * (
            105 + 105*g*r + 45*(g**2)*(r**2) + 10*(g**3)*(r**3) + (g**4)*(r**4)
        )

    def phiR(self, r):
        g = self.gamma
        return (-1) * np.exp(-g*r) * (15 + 15*g*r + 6*(g**2)*(r**2) + (g**3)*(r**3)) * (g**2)

    def phiRR(self, r):
        g = self.gamma
        return (g**4) * np.exp(-g*r) * (3 + 3*g*r + (g**2)*(r**2))


