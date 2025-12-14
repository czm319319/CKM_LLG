from typing import Optional, Iterable
import numpy as np

class Ridge:
    """
    Minimal ridge-regression helper used by the LLG/RMT code.

    Key idea:
      - Fit ridge for a *grid* of penalties z_grid, and do it efficiently by
        precomputing an eigen-decomposition once.
      - Supports both regimes:
          * "under": P <= T  (work in feature space via psi = X'X / T)
          * "over" : P >  T  (work in sample space via S = XX' / T)
    """


    def __init__(self, X, y, z_grid: Iterable[float]):
        # X: design/signals matrix (T x P). y: labels/target (T,)
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float).reshape(-1)
        self.n_samples, self.n_features = self.X.shape

        # z_grid: ridge strengths; must be strictly positive
        z_arr = np.atleast_1d(np.asarray(z_grid, dtype=float))
        if np.any(z_arr <= 0):
            raise ValueError("All ridge strengths in z_grid must be positive.")
        self.z_grid = z_arr
        self.M = self.z_grid.shape[0]


        if self.n_features <= self.n_samples:

            # Under-parameterized: use feature-space covariance psi = X'X / T
            self._regime = "under"
            self.psi = (self.X.T @ self.X) / self.n_samples

            # psi = Q diag(eigvals) Q'
            self._eigvals, self._Q = np.linalg.eigh(self.psi)
            # b = X^T y / T ; project to eigenbasis once
            self._b = (self.X.T @ self.y) / self.n_samples
            self._v = self._Q.T @ self._b
        else:
            # Over-parameterized: work in sample space S = XX' / T
            self._regime = "over"
            self.S = (self.X @ self.X.T) / self.n_samples
            self._eigvals, self._Q = np.linalg.eigh(self.S)

            # cache objects to avoid recomputing per-z solutions
            self._u = self._Q.T @ self.y
            self._XTQ = self.X.T @ self._Q


    def _betas_from_powers(self, power: int, scale: float) -> np.ndarray:
        """
        Internal workhorse to compute beta(z) for all z in z_grid in a vectorized way.

        It forms (zI + spectrum)^(-power) times precomputed projections, then maps back
        to the original coordinate system. `power=1` yields the standard ridge solution.
        """

        if self._regime == "under":
            d = self._eigvals[None, :]
            denom = self.z_grid[:, None] + d
            invp = denom ** (-power)

            W = invp * self._v[None, :]
            B = W @ self._Q.T
            return scale * B
        else:
            s = self._eigvals[None, :]
            denom = self.z_grid[:, None] + s
            invp = denom ** (-power)

            U = invp * self._u[None, :]
            B = (self._XTQ @ U.T).T
            return (scale / self.n_samples) * B

    def get_beta_hat(self) -> np.ndarray:
        """Return ridge coefficients for each z in z_grid. Shape: (M, P)."""
        return self._betas_from_powers(power=1, scale=+1.0)


    def pre_process_for_mse(self,
                            X_oos: Optional[np.ndarray] = None,
                            y_oos: Optional[np.ndarray] = None
                            ):
        """
        Shared prep for mse(): choose in-sample vs out-of-sample evaluation arrays,
        and compute beta(z) once.
        """
        if (X_oos is None) ^ (y_oos is None):
            raise ValueError("X_oos and y_oos must either both be None or both be provided.")

        X_eval = self.X if X_oos is None else np.asarray(X_oos, dtype=float)
        y_eval = self.y if y_oos is None else np.asarray(y_oos, dtype=float).reshape(-1)

        B = self.get_beta_hat()
        return X_eval, y_eval, B

    def mse(self,
            X_oos: Optional[np.ndarray] = None,
            y_oos: Optional[np.ndarray] = None,
            return_components: bool = False):
        """
        Vectorized MSE over z_grid.
        Returns shape (M,).

        If return_components=True, also returns a dict with basic decomposition pieces
        useful for debugging/plotting (y^2, yhat^2, interaction term, etc.).
        """
        X_eval, y_eval, B =self.pre_process_for_mse(X_oos=X_oos, y_oos=y_oos)

        Yhat = (X_eval @ B.T).T
        R = y_eval[None, :] - Yhat
        if not return_components:
            return np.mean(R ** 2, axis=1)
        else:
            components = {'y2': np.mean(y_eval[None, :] ** 2, axis=1),
                          'yhat2': np.mean(Yhat ** 2, axis=1),
                          'inteact': np.mean(-2 * y_eval[None, :] * Yhat, axis=1),
                          'beta': B,
                          'yhat': Yhat
                          }
            components['mse'] = components['y2'] + components['yhat2'] + components['inteact']
            return np.mean(R ** 2, axis=1), components
