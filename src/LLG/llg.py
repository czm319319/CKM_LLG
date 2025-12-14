import numpy as np
from src.utils.ridge import Ridge
from src.utils.rmt import RMT
from src.utils.r2_bounds import get_z_one_sided, r2_lower_bound_stats

class LLG_Bounds:
    """
    Scikit-learn style wrapper around Ridge + RMT utilities for LLG-based inference.

    Typical usage:
        model = LLG_Bounds(z_grid).fit(X_train, y_train)
        r2_oos = model.score(X_test, y_test)         # realized R2_OOS(z)
        llg    = model.llg(X_test)                   # LLG(z) computed from signals
        r2_oos, r2_lb, r2_ci = model.r2_bounds(X_test, y_test)  # bounds + CI
    """


    def __init__(self, z_grid, confidence_level=95):
        # z_grid: ridge penalties (vector). confidence_level: one-sided CI level in percent.
        self.r2_ci_lower_ = None
        self.r2_lb_ = None
        self.r2_oos_ = None
        self.llg_ = None
        self._is_fitted = None
        self.rmt_ = None
        self.ridge_ = None
        self.y_tr_ = None
        self.X_tr_ = None
        self.z_grid = np.asarray(z_grid, dtype=float)
        self.confidence_level = float(confidence_level)

    def fit(self, X, y):
        """
        Fit the ridge precomputations on training data and initialize the RMT/LLG machinery.
        """
        self.X_tr_ = np.asarray(X, dtype=float)
        self.y_tr_ = np.asarray(y, dtype=float).reshape(-1)

        self.ridge_ = Ridge(self.X_tr_, self.y_tr_, z_grid=self.z_grid)
        self.rmt_ = RMT(self.X_tr_, z_grid=self.z_grid)
        self._is_fitted = True
        return self

    def predict(self, X):
        """
        Return predictions for each z in z_grid.

        Output shape: (M, N) where M=len(z_grid), N=X.shape[0]
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        B = self.ridge_.get_beta_hat()
        yhat = (X @ B.T).T
        return yhat

    def score(self, X_test, y_test):
        """
        Return realized out-of-sample R^2(z) for each z in z_grid.
        """
        self._check_fitted()
        X_test = np.asarray(X_test, dtype=float)
        y_test = np.asarray(y_test, dtype=float).reshape(-1)

        mse_oos = self.ridge_.mse(X_oos=X_test, y_oos=y_test, return_components=False)
        r2_oos = 1.0 - mse_oos / np.mean(y_test ** 2.)
        self.r2_oos_ = r2_oos
        return r2_oos

    def llg(self, X_test):
        """
        Return LLG(z) for each z in z_grid.

        Note: This LLG estimator uses X_test through fancy_llg(), i.e., it depends only on
        signals (not on y_test), matching the “LLG depends only on S” intuition in the paper.
        """
        self._check_fitted()
        X_test = np.asarray(X_test, dtype=float)
        llg_arr = self.rmt_.fancy_llg(_ridge=self.ridge_, X_test=X_test)
        self.llg_ = llg_arr
        return llg_arr

    def r2_bounds(self, X_test, y_test):
        """
        Compute three arrays over z_grid:
          - R2_OOS(z): realized out-of-sample R^2
          - R2_LB(z):  asymptotic LLG-adjusted lower bound
          - CI_lower(z): one-sided CLT-adjusted lower confidence bound for population R^2_*
        """
        self._check_fitted()
        X_test = np.asarray(X_test, dtype=float)
        y_test = np.asarray(y_test, dtype=float).reshape(-1)

        if X_test.shape[1] != self.X_tr_.shape[1]:
            raise ValueError("X_test must have same number of columns as X used in fit().")

        z_one_sided = get_z_one_sided(self.confidence_level)

        r2_oos, r2_lb, lower = r2_lower_bound_stats(
            X_te=X_test,
            y_test=y_test,
            X_tr=self.X_tr_,
            ridge=self.ridge_,
            pivotal=self.rmt_,
            z_one_sided=z_one_sided,
        )

        self.r2_oos_, self.r2_lb_, self.r2_ci_lower_ = r2_oos, r2_lb, lower
        return r2_oos, r2_lb, lower

    def _check_fitted(self):
        """
        Private guard to mimic sklearn-style error messages.
        """
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError("Call fit(X, y) before using this method.")
