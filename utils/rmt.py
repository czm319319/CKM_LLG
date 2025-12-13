from typing import Iterable, Dict
import numpy as np

from .ridge import Ridge


class RMT:
    """
    Random-Matrix-Theory (RMT) utilities for the paper's LLG-based inference.

    Given a fitted Ridge object and out-of-sample signals X_test, this class computes:
      - LLG(z): the limits-to-learning gap L_b(z) (paper Eq. (24)/(28) specialized to ridge)
      - sigma_eps^2_hat(z): noise variance upper bound estimate via MSE_OOS / (1 + LLG)
      - auxiliary variance / quadratic-form terms used for asymptotic CI machinery
    """

    def __init__(self,
                 sig: np.ndarray,
                 z_grid: Iterable[float],
                 ):

        # z_grid: ridge penalties; must be strictly positive
        z_arr = np.atleast_1d(np.asarray(z_grid, dtype=float))
        if np.any(z_arr <= 0):
            raise ValueError("All shrinkage values in z_grid must be positive.")
        self.z_grid = z_arr

        # sig is the in-sample signal matrix S (shape T x P in the paper)
        self.S = sig
        self.T = sig.shape[0]
        self.P = sig.shape[1]
        self.c = float(self.P / self.T)  # statistical complexity c = P/T

    def _get_hat_sigma_V_sq(self, _ridge: Ridge, X_test: np.ndarray) -> np.ndarray:
        """
        Internal helper for estimating an asymptotic variance component used in CI formulas.
        Works in both regimes:
          - under-parameterized: P <= T (uses eigen-decomposition objects from Ridge._Q)
          - over-parameterized: P >  T (uses dual representation Ridge._XTQ)
        Returns an array over z_grid.
        """

        res = []
        if _ridge.n_features <= _ridge.n_samples:
            tmp = (X_test @ _ridge._Q)
            tmp = ((tmp.T @ tmp) / X_test.shape[0])  # empirical OOS covariance in eigenspace
            for ridge in _ridge.z_grid:
                # diagonal weights depending on eigenvalues and ridge penalty
                di = np.sqrt(_ridge._eigvals / ((ridge + _ridge._eigvals) ** 2))
                tmp_fin = tmp * di.reshape(-1, 1) * di.reshape(1, -1)
                res.append(2 * (tmp_fin ** 2).sum() / _ridge.n_samples)
        else:
            tmp = (X_test @ _ridge._XTQ)
            tmp = ((tmp.T @ tmp) / X_test.shape[0])
            for ridge in _ridge.z_grid:
                di = 1 / (ridge + _ridge._eigvals)
                tmp_fin = tmp * di.reshape(-1, 1) * di.reshape(1, -1)
                res.append(2 * (tmp_fin ** 2).sum() / (_ridge.n_samples ** 3))
        return np.array(res)

    def get_sigma_eps_2_hat(self,
                            _mse_oos: np.ndarray,
                            _ridge: Ridge = None,
                            X_test: np.ndarray=None):
        """
        Estimate sigma_eps^2 using the LLG correction:
          sigma_eps^2_hat(z) = MSE_OOS(z) / (1 + LLG(z))

        Returns:
          (sigma_eps_2_hat, llg_arr) both arrays over z_grid.
        """

        llg_new = self.fancy_llg(_ridge=_ridge,
                                 X_test=X_test)
        sigma_eps_2_hat =  _mse_oos / (1 + llg_new)
        return sigma_eps_2_hat, llg_new


    def fancy_llg(self,
                  _ridge: Ridge,
                  X_test: np.ndarray):
        """
        Compute the ridge-specialized LLG(z) using OOS signals X_test.
        This corresponds to the paper's L_b(z) evaluated with
        OOS signal covariance and ridge resolvent weights, implemented efficiently
        using precomputed Ridge linear-algebra factors.

        Returns an array over z_grid.
        """

        if _ridge.n_features <= _ridge.n_samples:
            # tmp approximates diagonal of (Q' X_test' X_test Q) / T_OOS in eigenspace
            tmp = ((X_test @ _ridge._Q) ** 2).mean(0)
            step1 =  (_ridge._eigvals.reshape(-1, 1) * ((_ridge.z_grid.reshape(1, -1) + _ridge._eigvals.reshape(-1, 1)) ** (-2)))
            return (step1 * tmp.reshape(-1, 1)).sum(0) / _ridge.n_samples
        else:
            tmp = ((X_test @ _ridge._XTQ) ** 2).mean(0)
            step1 = ((_ridge.z_grid.reshape(1, -1) + _ridge._eigvals.reshape(-1, 1)) ** (-2))
            result = (step1 * tmp.reshape(-1, 1)).sum(0) / (_ridge.n_samples ** 2)
            return result


    def _get_q_oos_norm_squared(self,
                                _ridge: Ridge,
                                X_test: np.ndarray,
                                y_test: np.ndarray
                                ):
        """
        Internal helper: computes a z-grid array of mean squared norms of an OOS
        quadratic-form quantity ("q_oos") built from prediction errors and ridge factors.
        Used as an ingredient in the paper's asymptotic variance / CI construction.

        Returns an array over z_grid.
        """

        difference = X_test @ _ridge.get_beta_hat().T - y_test.reshape(-1, 1)
        t_oos = X_test.shape[0]
        if _ridge.n_features <= _ridge.n_samples:
            step1 = 1 / (_ridge.z_grid.reshape(1, -1) + _ridge._eigvals.reshape(-1, 1)) * ((_ridge._Q.T @ X_test.T) @ difference)
            step2 = (_ridge.X @ _ridge._Q) @ step1 / t_oos
        else:

            step1 = _ridge._XTQ.T @ X_test.T @ difference / t_oos
            step2 = _ridge._Q @  (1 / (_ridge.z_grid.reshape(1, -1) + _ridge._eigvals.reshape(-1, 1)) * step1)
        answer = (step2 ** 2).mean(0)

        return answer
