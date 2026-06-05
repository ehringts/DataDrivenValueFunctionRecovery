"""Kernel classes for Hermite and control Gram matrices used by RKHS surrogates."""

import numpy as np
import abc


class Kernel(metaclass=abc.ABCMeta):
    """
    Radial basis kernel with k(x, y) = phi(r), where r = ||x - y||_2. The function phi is the radial
    profile, phiR denotes phi'(r)/r, and phiRR denotes (phiR)'(r)/r. These derivative factors are used
    to assemble value, gradient, mixed Hermite, and control Gram matrices.
    """

    def __init__(self, gamma):
        """Initialize the object and store the required parameters."""
        self.gamma = gamma

    def setGamma(self, gamma):
        """Set the kernel shape parameter gamma."""
        self.gamma = gamma

    @abc.abstractmethod
    def phi(self, r):
        """Evaluate the radial profile phi(r)."""

    @abc.abstractmethod
    def phiR(self, r):
        """Evaluate the radial derivative factor phi'(r) / r."""

    @abc.abstractmethod
    def phiRR(self, r):
        """Evaluate the second radial derivative factor (phiR)'(r) / r."""

    def getGramHermite(self, y, funcYList, x, funcXList):
        """Assemble the Hermite Gram matrix for value and derivative functionals."""
        y      = np.asarray(y)
        x      = np.asarray(x)
        fy     = np.asarray(funcYList, dtype=int)
        fx     = np.asarray(funcXList, dtype=int)
        ny, nx = (y.shape[1], x.shape[1])
        x2     = np.sum(x * x, axis=0)
        y2     = np.sum(y * y, axis=0)
        diff2  = y2[:, None] + x2[None, :] - 2.0 * (y.T @ x)
        diff   = np.sqrt(np.abs(diff2))
        K      = np.zeros((ny, nx), dtype=diff.dtype)
        iy0    = fy == 0
        ix0    = fx == 0
        iyp    = ~iy0
        ixp    = ~ix0
        if np.any(iy0) and np.any(ix0):
            idx    = np.ix_(iy0, ix0)
            K[idx] = self.phi(diff[idx])
        if np.any(ixp):
            cols_p = np.flatnonzero(ixp)
            fx_idx = fx[ixp] - 1
            x_fx   = x[fx_idx, cols_p]
            y_fx_T = y[fx_idx, :].T
            B_full = x_fx[None, :] - y_fx_T
        if np.any(iyp):
            rows_p = np.flatnonzero(iyp)
            fy_idx = fy[iyp] - 1
            y_fy   = y[fy_idx, rows_p]
            x_fy   = x[fy_idx, :]
            A_full = y_fy[:, None] - x_fy
        if np.any(iy0) and np.any(ixp):
            idx    = np.ix_(iy0, ixp)
            K[idx] = self.phiR(diff[idx]) * B_full[iy0, :]
        if np.any(iyp) and np.any(ix0):
            idx    = np.ix_(iyp, ix0)
            K[idx] = self.phiR(diff[idx]) * A_full[:, ix0]
        if np.any(iyp) and np.any(ixp):
            rows_p = np.flatnonzero(iyp)
            cols_p = np.flatnonzero(ixp)
            idx    = np.ix_(iyp, ixp)
            A      = A_full[:, ixp]
            B      = B_full[rows_p, :]
            K[idx] = self.phiRR(diff[idx]) * (A * B)
            eq     = fy[iyp][:, None] == fx[ixp][None, :]
            K[idx] -= self.phiR(diff[idx]) * eq
        return K

    def getGramControl(self, y, funcYList, x, funcXList, R, g):
        """Assemble the control Gram matrix for direct feedback-control functionals."""
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        d, Ny  = y.shape
        d2, Nx = x.shape
        if d != d2:
            raise ValueError('y and x must have the same state dimension.')
        funcY = np.asarray(funcYList, dtype=int).reshape(-1)
        funcX = np.asarray(funcXList, dtype=int).reshape(-1)

        def batch_g(Z):
            """Normalize batched evaluations of the control matrix to one common array format."""
            Z     = np.asarray(Z, dtype=float)
            dz, N = Z.shape
            G     = np.asarray(g(Z), dtype=float)
            if G.ndim == 1:
                return G.reshape(dz, 1, 1)
            if G.ndim == 2:
                if G.shape[0] != dz:
                    raise ValueError('g(Z) returned 2D array with wrong first dimension.')
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
            raise ValueError('g(Z) has unsupported shape.')
        GY = batch_g(y)
        GX = batch_g(x)
        m  = GY.shape[1]
        R  = np.asarray(R, dtype=float)
        if R.ndim == 0:
            Rmat = float(R) * np.eye(m)
        elif R.ndim == 1:
            Rmat = np.diag(R)
        else:
            Rmat = R
        BY    = GY.transpose(1, 0, 2).reshape(m, d * Ny)
        BX    = GX.transpose(1, 0, 2).reshape(m, d * Nx)
        AallY = -0.5 * np.linalg.solve(Rmat, BY)
        AallX = -0.5 * np.linalg.solve(Rmat, BX)
        AallY = AallY.reshape(m, d, Ny).transpose(2, 0, 1)
        AallX = AallX.reshape(m, d, Nx).transpose(2, 0, 1)
        compY = funcY - 1
        compX = funcX - 1
        AY    = AallY[np.arange(Ny), compY, :]
        AX    = AallX[np.arange(Nx), compX, :]
        yT    = y.T
        xT    = x.T
        yy    = np.sum(y * y, axis=0)[:, None]
        xx    = np.sum(x * x, axis=0)[None, :]
        sq    = yy + xx - 2.0 * (yT @ x)
        sq    = np.maximum(sq, 0.0)
        r     = np.sqrt(sq)
        phiR  = self.phiR(r)
        phiRR = self.phiRR(r)
        AY_x  = AY @ x
        AY_y  = np.sum(AY * yT, axis=1)[:, None]
        AY_h  = AY_x - AY_y
        AX_x  = np.sum(AX * xT, axis=1)[None, :]
        AX_y  = yT @ AX.T
        AX_h  = AX_x - AX_y
        AY_AX = AY @ AX.T
        K     = -phiRR * AY_h * AX_h - phiR * AY_AX
        return K


class QuadWendland(Kernel):
    """Compactly supported quadratic Wendland radial basis kernel."""

    def __init__(self, gamma, d):
        """Initialize the object and store the required parameters."""
        self.l     = np.floor(d / 2) + 2 + 1
        self.gamma = gamma

    def phi(self, r):
        """Evaluate the radial profile phi(r)."""
        return (self.gamma * r <= 1) * (1 - self.gamma * r) ** (self.l + 2) * ((self.l ** 2 + 4 * self.l + 3) * (self.gamma * r) ** 2 + (3 * self.l + 6) * self.gamma * r + 3)

    def phiR(self, r):
        """Evaluate the radial derivative factor phi'(r) / r."""
        return (self.gamma * r <= 1) * (1 - self.gamma * r) ** (self.l + 1) * -self.gamma ** 2 * (12 + 7 * self.l + self.l ** 2) * (1 + (1 + self.l) * self.gamma * r)

    def phiRR(self, r):
        """Evaluate the second radial derivative factor (phiR)'(r) / r."""
        return (self.gamma * r <= 1) * (1 - self.gamma * r) ** self.l * self.gamma ** 4 * (24 + 50 * self.l + 35 * self.l ** 2 + 10 * self.l ** 3 + self.l ** 4)


class QuadMatern(Kernel):
    """Quadratic Matern radial basis kernel with phi(r) = exp(-gamma*r) * (3 + 3*gamma*r + gamma^2*r^2)."""

    def phi(self, r):
        """Evaluate the radial profile phi(r)."""
        return np.exp(-self.gamma * r) * (3 + 3 * self.gamma * r + self.gamma ** 2 * r ** 2)

    def phiR(self, r):
        """Evaluate the radial derivative factor phi'(r) / r."""
        return -1 * np.exp(-self.gamma * r) * (1 + self.gamma * r) * self.gamma ** 2

    def phiRR(self, r):
        """Evaluate the second radial derivative factor (phiR)'(r) / r."""
        return self.gamma ** 4 * np.exp(-self.gamma * r)


class Gauss(Kernel):
    """Gaussian radial basis kernel with phi(r) = exp(-gamma*r^2)."""

    def phi(self, r):
        """Evaluate the radial profile phi(r)."""
        return np.exp(-self.gamma * r ** 2)

    def phiR(self, r):
        """Evaluate the radial derivative factor phi'(r) / r."""
        return -2 * self.gamma * np.exp(-self.gamma * r ** 2)

    def phiRR(self, r):
        """Evaluate the second radial derivative factor (phiR)'(r) / r."""
        return 4 * self.gamma ** 2 * np.exp(-self.gamma * r ** 2)


class InvMulti(Kernel):
    """Inverse multiquadric radial basis kernel with phi(r) = 1 / sqrt(1 + gamma*r^2)."""

    def phi(self, r):
        """Evaluate the radial profile phi(r)."""
        return 1 / np.sqrt(1 + self.gamma * r ** 2)

    def phiR(self, r):
        """Evaluate the radial derivative factor phi'(r) / r."""
        return -1 * self.gamma * (1 / np.sqrt((1 + self.gamma * r ** 2) ** 3))

    def phiRR(self, r):
        """Evaluate the second radial derivative factor (phiR)'(r) / r."""
        return 3 * self.gamma ** 2 * (1 / np.sqrt((1 + self.gamma * r ** 2) ** 5))


class LinMatern(Kernel):
    """Linear Matern radial basis kernel with phi(r) = exp(-gamma*r) * (1 + gamma*r)."""

    def phi(self, r):
        """Evaluate the radial profile phi(r)."""
        return np.exp(-self.gamma * r) * (1 + self.gamma * r)

    def phiR(self, r):
        """Evaluate the radial derivative factor phi'(r) / r."""
        return -1 * np.exp(-self.gamma * r) * self.gamma ** 2

    def phiRR(self, r):
        """Evaluate the second radial derivative factor (phiR)'(r) / r."""
        diffMask1 = r < 10 ** (-14)
        diffMask2 = r > 10 ** (-14)
        return self.gamma ** 3 * np.exp(-self.gamma * r) * (1 / (r + diffMask1)) * diffMask2


class CubMatern(Kernel):
    """Cubic Matern radial basis kernel with an exponential radial factor and a third-order polynomial."""

    def phi(self, r):
        """Evaluate the radial profile phi(r)."""
        g = self.gamma
        return np.exp(-g * r) * (15 + 15 * g * r + 6 * g ** 2 * r ** 2 + g ** 3 * r ** 3)

    def phiR(self, r):
        """Evaluate the radial derivative factor phi'(r) / r."""
        g = self.gamma
        return -1 * np.exp(-g * r) * (3 + 3 * g * r + g ** 2 * r ** 2) * g ** 2

    def phiRR(self, r):
        """Evaluate the second radial derivative factor (phiR)'(r) / r."""
        g = self.gamma
        return g ** 4 * np.exp(-g * r) * (1 + g * r)


class QuartMatern(Kernel):
    """Quartic Matern radial basis kernel with an exponential radial factor and a fourth-order polynomial."""

    def phi(self, r):
        """Evaluate the radial profile phi(r)."""
        g = self.gamma
        return np.exp(-g * r) * (105 + 105 * g * r + 45 * g ** 2 * r ** 2 + 10 * g ** 3 * r ** 3 + g ** 4 * r ** 4)

    def phiR(self, r):
        """Evaluate the radial derivative factor phi'(r) / r."""
        g = self.gamma
        return -1 * np.exp(-g * r) * (15 + 15 * g * r + 6 * g ** 2 * r ** 2 + g ** 3 * r ** 3) * g ** 2

    def phiRR(self, r):
        """Evaluate the second radial derivative factor (phiR)'(r) / r."""
        g = self.gamma
        return g ** 4 * np.exp(-g * r) * (3 + 3 * g * r + g ** 2 * r ** 2)


class KernelProduct(metaclass=abc.ABCMeta):
    """
    Radial product kernel with k(x, y) = phi(r) * (x^T y)^2, where r = ||x - y||_2. The function phi is
    the radial profile, phiR denotes phi'(r)/r, and phiRR denotes (phiR)'(r)/r. The polynomial factor
    (x^T y)^2 is included in the Hermite and control Gram formulas.
    """

    def __init__(self, gamma):
        """Initialize the object and store the required parameters."""
        self.gamma = gamma

    def setGamma(self, gamma):
        """Set the kernel shape parameter gamma."""
        self.gamma = gamma

    @abc.abstractmethod
    def phi(self, r):
        """Evaluate the radial profile phi(r)."""

    @abc.abstractmethod
    def phiR(self, r):
        """Evaluate the radial derivative factor phi'(r) / r."""

    @abc.abstractmethod
    def phiRR(self, r):
        """Evaluate the second radial derivative factor (phiR)'(r) / r."""

    def getGramHermite(self, y, funcYList, x, funcXList):
        """Assemble the Hermite Gram matrix for value and derivative functionals."""
        y      = np.asarray(y)
        x      = np.asarray(x)
        funcY  = np.asarray(funcYList, dtype=np.int64)
        funcX  = np.asarray(funcXList, dtype=np.int64)
        d, Ny  = y.shape
        d2, Nx = x.shape
        if d != d2:
            raise ValueError('x and y must have the same first dimension (ambient dimension).')
        if funcY.shape[0] != Ny or funcX.shape[0] != Nx:
            raise ValueError('funcYList length must be Ny and funcXList length must be Nx.')
        if np.any(funcY < 0) or np.any(funcY > d) or np.any(funcX < 0) or np.any(funcX > d):
            raise ValueError('Functional indices must be in {0,1,...,d} (1-based for derivatives).')
        xx          = np.sum(x * x, axis=0, keepdims=True)
        yy          = np.sum(y * y, axis=0, keepdims=True).T
        sq          = yy + xx - 2.0 * (y.T @ x)
        sq          = np.maximum(sq, 0.0)
        r           = np.sqrt(sq)
        s           = y.T @ x
        phi         = self.phi(r)
        phiR        = self.phiR(r)
        phiRR       = self.phiRR(r)
        fy          = funcY[:, None]
        fx          = funcX[None, :]
        m00         = (fy == 0) & (fx == 0)
        m0x         = (fy == 0) & (fx > 0)
        my0         = (fy > 0) & (fx == 0)
        myx         = (fy > 0) & (fx > 0)
        p_cols_safe = np.where(funcX > 0, funcX - 1, 0)
        q_rows_safe = np.where(funcY > 0, funcY - 1, 0)
        j           = np.arange(Nx)
        i           = np.arange(Ny)
        x_p         = x[p_cols_safe, j]
        y_p         = y[p_cols_safe, :].T
        y_q         = y[q_rows_safe, i]
        x_q         = x[q_rows_safe, :]
        dp          = x_p[None, :] - y_p
        dq          = x_q - y_q[:, None]
        delta       = q_rows_safe[:, None] == p_cols_safe[None, :]
        K           = np.zeros((Ny, Nx), dtype=np.result_type(phi, s, r))
        s2          = s * s
        K[m00]      = (phi * s2)[m00]
        tmp0x       = s2 * phiR * dp + 2.0 * phi * s * y_p
        K[m0x]      = tmp0x[m0x]
        tmpy0       = -s2 * phiR * dq + 2.0 * phi * s * x_q
        K[my0]      = tmpy0[my0]
        tmpyx       = -s2 * phiRR * dp * dq - s2 * phiR * delta + 2.0 * s * phiR * (dp * x_q - dq * y_p) + 2.0 * phi * x_q * y_p + 2.0 * phi * s * delta
        K[myx]      = tmpyx[myx]
        return K

    def getGramControl(self, y, funcYList, x, funcXList, R, g):
        """Assemble the control Gram matrix for direct feedback-control functionals."""
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        d, Ny  = y.shape
        d2, Nx = x.shape
        if d != d2:
            raise ValueError('y and x must have the same state dimension.')
        funcY = np.asarray(funcYList, dtype=int).reshape(-1)
        funcX = np.asarray(funcXList, dtype=int).reshape(-1)
        if funcY.size != Ny:
            raise ValueError('funcYList must have length y.shape[1].')
        if funcX.size != Nx:
            raise ValueError('funcXList must have length x.shape[1].')

        def batch_g(Z):
            """Normalize batched evaluations of the control matrix to one common array format."""
            Z     = np.asarray(Z, dtype=float)
            dz, N = Z.shape
            G     = np.asarray(g(Z), dtype=float)
            if G.ndim == 1:
                if G.size != dz:
                    raise ValueError('g(Z) returned vector with wrong length.')
                return G.reshape(dz, 1, 1)
            if G.ndim == 2:
                if G.shape[0] != dz:
                    raise ValueError('g(Z) returned 2D array with wrong first dimension.')
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
            raise ValueError('g(Z) must return shape (d,m), (d,m,N), (N,d,m), (d,N,m), or (d,N).')
        GY = batch_g(y)
        GX = batch_g(x)
        m  = GY.shape[1]
        if GX.shape[1] != m:
            raise ValueError('g(y) and g(x) have different control dimensions.')
        R = np.asarray(R, dtype=float)
        if R.ndim == 0:
            Rmat = float(R) * np.eye(m)
        elif R.ndim == 1:
            if R.size != m:
                raise ValueError('R vector has wrong length.')
            Rmat = np.diag(R)
        elif R.ndim == 2:
            if R.shape != (m, m):
                raise ValueError('R matrix has wrong shape.')
            Rmat = R
        else:
            raise ValueError('R must be scalar, vector, or matrix.')
        BY    = GY.transpose(1, 0, 2).reshape(m, d * Ny)
        BX    = GX.transpose(1, 0, 2).reshape(m, d * Nx)
        AallY = -0.5 * np.linalg.solve(Rmat, BY)
        AallX = -0.5 * np.linalg.solve(Rmat, BX)
        AallY = AallY.reshape(m, d, Ny).transpose(2, 0, 1)
        AallX = AallX.reshape(m, d, Nx).transpose(2, 0, 1)
        compY = funcY - 1
        compX = funcX - 1
        if np.any(compY < 0) or np.any(compY >= m):
            raise ValueError('funcYList contains invalid control components.')
        if np.any(compX < 0) or np.any(compX >= m):
            raise ValueError('funcXList contains invalid control components.')
        AY    = AallY[np.arange(Ny), compY, :]
        AX    = AallX[np.arange(Nx), compX, :]
        yT    = y.T
        xT    = x.T
        yy    = np.sum(y * y, axis=0)[:, None]
        xx    = np.sum(x * x, axis=0)[None, :]
        sq    = yy + xx - 2.0 * (yT @ x)
        sq    = np.maximum(sq, 0.0)
        r     = np.sqrt(sq)
        s     = yT @ x
        phi   = self.phi(r)
        phiR  = self.phiR(r)
        phiRR = self.phiRR(r)
        s2    = s * s
        AY_x  = AY @ x
        AY_y  = np.sum(AY * yT, axis=1)[:, None]
        AY_h  = AY_x - AY_y
        AX_x  = np.sum(AX * xT, axis=1)[None, :]
        AX_y  = yT @ AX.T
        AX_h  = AX_x - AX_y
        AY_AX = AY @ AX.T
        K     = -s2 * phiRR * AY_h * AX_h - s2 * phiR * AY_AX + 2.0 * s * phiR * (AX_h * AY_x - AY_h * AX_y) + 2.0 * phi * AY_x * AX_y + 2.0 * phi * s * AY_AX
        return K


class QuadWendlandProduct(KernelProduct):
    """Product version of the compactly supported quadratic Wendland kernel."""

    def __init__(self, gamma, d, case=0):
        """Initialize the object and store the required parameters."""
        self.l     = np.floor(d / 2) + 2 + 1
        self.case  = case
        self.gamma = gamma

    def phi(self, r):
        """Evaluate the radial profile phi(r)."""
        return (self.gamma * r <= 1) * (1 - self.gamma * r) ** (self.l + 2) * ((self.l ** 2 + 4 * self.l + 3) * (self.gamma * r) ** 2 + (3 * self.l + 6) * self.gamma * r + 3)

    def phiR(self, r):
        """Evaluate the radial derivative factor phi'(r) / r."""
        return (self.gamma * r <= 1) * (1 - self.gamma * r) ** (self.l + 1) * -self.gamma ** 2 * (12 + 7 * self.l + self.l ** 2) * (1 + (1 + self.l) * self.gamma * r)

    def phiRR(self, r):
        """Evaluate the second radial derivative factor (phiR)'(r) / r."""
        return (self.gamma * r <= 1) * (1 - self.gamma * r) ** self.l * self.gamma ** 4 * (24 + 50 * self.l + 35 * self.l ** 2 + 10 * self.l ** 3 + self.l ** 4)


class QuadMaternProduct(KernelProduct):
    """Product version of the quadratic Matern kernel."""

    def phi(self, r):
        """Evaluate the radial profile phi(r)."""
        return np.exp(-self.gamma * r) * (3 + 3 * self.gamma * r + self.gamma ** 2 * r ** 2)

    def phiR(self, r):
        """Evaluate the radial derivative factor phi'(r) / r."""
        return -1 * np.exp(-self.gamma * r) * (1 + self.gamma * r) * self.gamma ** 2

    def phiRR(self, r):
        """Evaluate the second radial derivative factor (phiR)'(r) / r."""
        return self.gamma ** 4 * np.exp(-self.gamma * r)


class GaussProduct(KernelProduct):
    """Product version of the Gaussian kernel."""

    def phi(self, r):
        """Evaluate the radial profile phi(r)."""
        return np.exp(-self.gamma * r ** 2)

    def phiR(self, r):
        """Evaluate the radial derivative factor phi'(r) / r."""
        return -2 * self.gamma * np.exp(-self.gamma * r ** 2)

    def phiRR(self, r):
        """Evaluate the second radial derivative factor (phiR)'(r) / r."""
        return 4 * self.gamma ** 2 * np.exp(-self.gamma * r ** 2)


class InvMultiProduct(KernelProduct):
    """Product version of the inverse multiquadric kernel."""

    def phi(self, r):
        """Evaluate the radial profile phi(r)."""
        return 1 / np.sqrt(1 + self.gamma * r ** 2)

    def phiR(self, r):
        """Evaluate the radial derivative factor phi'(r) / r."""
        return -1 * self.gamma * (1 / np.sqrt((1 + self.gamma * r ** 2) ** 3))

    def phiRR(self, r):
        """Evaluate the second radial derivative factor (phiR)'(r) / r."""
        return 3 * self.gamma ** 2 * (1 / np.sqrt((1 + self.gamma * r ** 2) ** 5))


class LinMaternProduct(KernelProduct):
    """Product version of the linear Matern kernel."""

    def phi(self, r):
        """Evaluate the radial profile phi(r)."""
        return np.exp(-self.gamma * r) * (1 + self.gamma * r)

    def phiR(self, r):
        """Evaluate the radial derivative factor phi'(r) / r."""
        return -1 * np.exp(-self.gamma * r) * self.gamma ** 2

    def phiRR(self, r):
        """Evaluate the second radial derivative factor (phiR)'(r) / r."""
        diffMask1 = r < 10 ** (-14)
        diffMask2 = r > 10 ** (-14)
        return self.gamma ** 3 * np.exp(-self.gamma * r) * (1 / (r + diffMask1)) * diffMask2


class CubMaternProduct(KernelProduct):
    """Product version of the cubic Matern kernel."""

    def phi(self, r):
        """Evaluate the radial profile phi(r)."""
        g = self.gamma
        return np.exp(-g * r) * (15 + 15 * g * r + 6 * g ** 2 * r ** 2 + g ** 3 * r ** 3)

    def phiR(self, r):
        """Evaluate the radial derivative factor phi'(r) / r."""
        g = self.gamma
        return -1 * np.exp(-g * r) * (3 + 3 * g * r + g ** 2 * r ** 2) * g ** 2

    def phiRR(self, r):
        """Evaluate the second radial derivative factor (phiR)'(r) / r."""
        g = self.gamma
        return g ** 4 * np.exp(-g * r) * (1 + g * r)


class QuartMaternProduct(KernelProduct):
    """Product version of the quartic Matern kernel."""

    def phi(self, r):
        """Evaluate the radial profile phi(r)."""
        g = self.gamma
        return np.exp(-g * r) * (105 + 105 * g * r + 45 * g ** 2 * r ** 2 + 10 * g ** 3 * r ** 3 + g ** 4 * r ** 4)

    def phiR(self, r):
        """Evaluate the radial derivative factor phi'(r) / r."""
        g = self.gamma
        return -1 * np.exp(-g * r) * (15 + 15 * g * r + 6 * g ** 2 * r ** 2 + g ** 3 * r ** 3) * g ** 2

    def phiRR(self, r):
        """Evaluate the second radial derivative factor (phiR)'(r) / r."""
        g = self.gamma
        return g ** 4 * np.exp(-g * r) * (3 + 3 * g * r + g ** 2 * r ** 2)
