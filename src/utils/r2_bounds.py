import numpy as np
from scipy.stats import norm
from .ridge import Ridge
from .rmt import RMT


def get_z_one_sided(confidence_level: float) -> float:
    """
    Convenience: convert a one-sided confidence level (e.g., 95) into the corresponding
    standard-normal quantile z (e.g., 1.645).
    """
    alpha = 1 - confidence_level / 100.0
    z_one_sided = norm.ppf(1 - alpha)

    return z_one_sided


def r2_lower_bound_stats(
        X_te: np.ndarray,
        y_test: np.ndarray,
        X_tr: np.ndarray,
        ridge: Ridge,
        pivotal: RMT,
        z_one_sided: float = 1.645,
):
    """
    Public-facing wrapper used in the tutorial:

    Given:
      - fitted Ridge model (trained on X_tr, y_tr inside ridge)
      - RMT pivotal object (constructed from in-sample signals, with same z_grid)
      - out-of-sample test set (X_te, y_test)

    Returns three arrays over z_grid:
      1) r2_oos: realized out-of-sample R^2(z)
      2) r2_lb:  asymptotic LLG-adjusted lower bound (paper Eq. (25)/(30)-style)
      3) lower_bound: one-sided CLT-adjusted 95% (or chosen) lower CI for R^2_*
    """

    # Realized out-of-sample MSE and R^2 for each ridge penalty z
    mse_oos = ridge.mse(X_oos=X_te, y_oos=y_test, return_components=False)
    r2_oos = 1.0 - mse_oos / np.mean(y_test ** 2)

    # LLG(z) and implied upper bound on sigma_eps^2 via sigma^2 <= MSE_OOS / (1 + LLG)
    sigma_eps_sq_upper_bound, llg_arr = pivotal.get_sigma_eps_2_hat(mse_oos, _ridge=ridge, X_test=X_te)
    sigma_eps_sq_best_upper_bound = np.min(sigma_eps_sq_upper_bound)

    # Auxiliary terms for the asymptotic variance bound used in the CI
    sigma2_V = pivotal.get_hat_sigma_V_sq(_ridge=ridge, X_test=X_te)
    q_oos_norm_sqrd = pivotal.get_q_oos_norm_squared(_ridge=ridge,
                                                     X_test=X_te,
                                                     y_test=y_test
                                                     )

    # Combine everything into the simplified lower bound + CI computation
    r_oos, r2_lb, lower_bound = r2_lower_bound_simplified(T=X_tr.shape[0],
                                                      Toos=X_te.shape[0],
                                                      _ridge=ridge,
                                                      mse_oos_arr=mse_oos,
                                                      mse_0=np.mean(y_test ** 2),
                                                      llg_arr=llg_arr,
                                                      q_oos_norm_sqrd=q_oos_norm_sqrd,
                                                      sigma2_V_hat_arr=sigma2_V,
                                                      z_one_sided=z_one_sided,
                                                      X_test=X_te,
                                                      y_test=y_test,
                                                      sigma_eps2_best_upper_bound=sigma_eps_sq_best_upper_bound
                                                      )
    return r2_oos, r2_lb, lower_bound


def polynom(x, a1, a2):
    """Utility: quadratic polynomial a1*x + a2*x^2."""
    return a1 * x + a2 * x ** 2


def a1_coefficient(mse0,
                   t_oos,
                   t_,
                   q_oos_norm_sqrd,
                   mse_oos,
                   _ridge: Ridge,
                   X_oos: np.ndarray,
                   y_oos: np.ndarray
                   ):
    """
    Internal coefficient used in an upper bound on the asymptotic variance of the
    LLG-adjusted R^2 lower bound.

    This function only depends on objects already computed in the pipeline
    (MSE_OOS, q_oos term, etc.). It is written vectorized over z_grid.
    """
    beta_hat = _ridge.get_beta_hat()
    Yhat = (X_oos @ beta_hat.T)
    beta_psi_betahat = y_oos.reshape(1, -1) @ Yhat / X_oos.shape[0]
    term1 = mse0 ** 2 * (t_oos / t_) * (4 * q_oos_norm_sqrd + 4 * (t_oos / t_) * mse_oos)
    term2 = 4 * (mse_oos ** 2) * mse0 - 8 * mse0 * mse_oos * (mse0 - beta_psi_betahat)
    return term1 + term2


def a2_coefficient(mse0,
                   t_oos,
                   t_,
                   mse_oos,
                   _ridge: Ridge,
                   sigma2_V_hat_arr: np.ndarray,
                   llg_arr: np.ndarray
                   ):
    """
    Internal coefficient used alongside a1_coefficient to form a conservative variance
    upper bound (quadratic in sigma_eps^2). Vectorized over z_grid.
    """
    term1 = mse0 ** 2 * (t_oos / t_) * (2 * (t_ / t_oos) - sigma2_V_hat_arr - 4 * (t_ / t_oos) * (1 + llg_arr))
    term2 = -2 * (mse_oos ** 2) + 4 * mse0 * mse_oos
    return term1 + term2


def r2_lower_bound_simplified(
        T: int,
        Toos: int,
        _ridge: Ridge,
        mse_oos_arr: np.ndarray,
        mse_0: float,
        X_test: np.ndarray,
        y_test: np.ndarray,
        llg_arr: np.ndarray,
        q_oos_norm_sqrd: np.ndarray,
        sigma2_V_hat_arr: np.ndarray,
        sigma_eps2_best_upper_bound: float,
        z_one_sided: float = 1.645,  # 95% one-sided normal quantile
):
    """
    Core computation for:
      - r2_oos(z): realized OOS R^2
      - r2_lb(z):  LLG-adjusted asymptotic lower bound (r2_oos + LLG)/(1+LLG)
      - lower_bound(z): one-sided CLT-adjusted lower confidence bound for R^2_*

    Notes for users:
      - This is intentionally written as a minimal “plumbing” function.
      - It uses a conservative upper bound on Var( R^2 lower bound ), then applies
        a normal quantile z_one_sided / sqrt(Toos).
    """

    a1 = a1_coefficient(mse0=mse_0,
                        t_oos=Toos,
                        t_=T,
                        q_oos_norm_sqrd=q_oos_norm_sqrd,
                        mse_oos=mse_oos_arr,
                        _ridge=_ridge,
                        X_oos=X_test,
                        y_oos=y_test)

    a2 = a2_coefficient(mse0=mse_0,
                        t_oos=Toos,
                        t_=T,
                        mse_oos=mse_oos_arr,
                        _ridge=_ridge,
                        llg_arr=llg_arr,
                        sigma2_V_hat_arr=sigma2_V_hat_arr)

    # Choose the tighter of two quadratic-based candidates (details in paper appendix)
    candidate1 = polynom(sigma_eps2_best_upper_bound, a1, a2)
    opt = (-0.5 * a1 / a2)
    opt = opt * (opt > 0)
    candidate2 = polynom(opt, a1, a2)

    upper_bound_on_sigma2_r2 = np.where(candidate1 > candidate2, candidate1, candidate2) / (1 + llg_arr) ** 2 / (
                mse_0 ** 4)
    upper_bound_on_sigma2_r2 = upper_bound_on_sigma2_r2 * (upper_bound_on_sigma2_r2 > 0)

    # Assemble R^2 objects
    r2_oos = 1 - mse_oos_arr / mse_0
    r2_lb = (r2_oos + llg_arr) / (1 + llg_arr)
    lower_bound = r2_lb - z_one_sided * np.sqrt(upper_bound_on_sigma2_r2) / np.sqrt(Toos)
    return r2_oos, r2_lb, lower_bound
