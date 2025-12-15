import numpy as np
import numba as nb


@nb.njit(fastmath=True)
def ewma_ex_ante_risk(
    weights: np.ndarray,          # (T, N)
    returns: np.ndarray,          # (T, N)
    half_life: float,
    shrinkage: float,
    horizon_steps: np.ndarray,    # (T,) K_k
    var_floor: float = 0.0,
) -> np.ndarray:
    T, N = weights.shape
    lam = 2.0 ** (-1.0 / half_life)

    var_port = 0.0
    var_asset = np.zeros(N)
    out = np.empty(T)

    for t in range(T):
        wt = weights[t]
        rt = returns[t]

        pnl = 0.0
        ridge = 0.0
        valid = False

        # asset EWMA
        for i in range(N):
            ri = rt[i]
            if not np.isnan(ri):
                var_asset[i] = lam * var_asset[i] + (1.0 - lam) * ri * ri

        # portfolio quadratic forms
        for i in range(N):
            wi = wt[i]
            ri = rt[i]
            if not np.isnan(wi) and not np.isnan(ri):
                pnl += wi * ri
                ridge += wi * wi * var_asset[i]
                valid = True

        if valid:
            var_port = lam * var_port + (1.0 - lam) * pnl * pnl

        var_ret = (1.0 - shrinkage) * var_port + shrinkage * ridge
        var_step = var_ret / horizon_steps[t]

        if var_step < var_floor:
            var_step = var_floor

        out[t] = np.sqrt(var_step)

    return out
    
    
@nb.njit(fastmath=True)
def risk_target_positions(
    raw_weights: np.ndarray,
    ex_ante_risk: np.ndarray,
    target_risk: float,
    smooth: float = 1.0,
) -> np.ndarray:
    T, N = raw_weights.shape
    out = np.empty_like(raw_weights)

    kappa_prev = 1.0

    for t in range(T):
        sigma = ex_ante_risk[t]
        if sigma > 0.0:
            kappa = target_risk / sigma
        else:
            kappa = kappa_prev

        kappa = (1.0 - smooth) * kappa_prev + smooth * kappa

        for i in range(N):
            out[t, i] = kappa * raw_weights[t, i]

        kappa_prev = kappa

    return out


def estimate_alpha_half_life(
    signal: np.ndarray,
    returns: np.ndarray,
    max_lag: int,
    min_corr: float = 1e-4,
    conf_level: float = 0.95,
):
    """
    Estimates alpha half-life from signal/return cross-decay and
    returns confidence intervals.

    Parameters
    ----------
    signal : (T,) array
        Trading signal s_t
    returns : (T,) array
        Returns r_t
    max_lag : int
        Maximum forward lag k
    min_corr : float
        Minimum |corr| to include in regression
    conf_level : float
        Confidence level for CI

    Returns
    -------
    hl_hat : float
        Point estimate of alpha half-life
    hl_ci : (float, float)
        Lower / upper confidence bounds
    details : dict
        Regression diagnostics
    """
    T = len(signal)
    lags = []
    ys = []

    for k in range(1, max_lag + 1):
        x = signal[:-k]
        y = returns[k:]

        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 10:
            continue

        corr = np.corrcoef(x[mask], y[mask])[0, 1]
        if np.abs(corr) < min_corr:
            continue

        lags.append(k)
        ys.append(np.log(np.abs(corr)))

    if len(lags) < 3:
        raise ValueError("Not enough significant lags to estimate half-life.")

    X = np.column_stack([np.ones(len(lags)), np.asarray(lags)])
    y = np.asarray(ys)

    # OLS
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a_hat, b_hat = beta

    # Residual variance
    resid = y - X @ beta
    sigma2 = resid @ resid / (len(y) - 2)

    # Covariance of beta
    XtX_inv = np.linalg.inv(X.T @ X)
    var_b = sigma2 * XtX_inv[1, 1]
    se_b = np.sqrt(var_b)

    # CI for b
    z = abs(np.quantile(np.random.normal(size=100_000), (1 - conf_level) / 2))
    b_lo = b_hat - z * se_b
    b_hi = b_hat + z * se_b

    # Map to half-life (note sign!)
    log2 = np.log(0.5)
    hl_hat = log2 / b_hat
    hl_lo = log2 / b_hi
    hl_hi = log2 / b_lo

    return hl_hat, (hl_lo, hl_hi), {
        "phi_hat": np.exp(b_hat),
        "b_hat": b_hat,
        "se_b": se_b,
        "lags_used": np.asarray(lags),
        "corr_logs": y,
    }
