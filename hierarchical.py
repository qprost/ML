# pls2_unified.py
"""
Unified PLS2 module with:
 - PLS2 estimator (whitening, RMT cleaning, optional sparse numba FISTA)
 - Intraday-only label builder (exclude horizons crossing session boundary)
 - Volatility-normalized target builder
 - Purged + Embargoed CV splitter
 - Ridge tuning helper (purged CV)
 - Demo: end-to-end synthetic intraday example

Save as pls2_unified.py and import PLS2, build_intraday_label, make_vol_normalized_targets,
PurgedWalkForwardSplit, tune_ridge_via_cv, demo_end_to_end.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted

# ----------------------------
# Numba kernels (soft-threshold, strict FISTA inner loop)
# ----------------------------


@njit
def _soft_threshold_jitted(a: np.ndarray, lam: float) -> np.ndarray:
    out = np.empty_like(a)
    for i in range(a.size):
        v = a[i]
        if v > lam:
            out[i] = v - lam
        elif v < -lam:
            out[i] = v + lam
        else:
            out[i] = 0.0
    return out


@njit
def _fista_inner_strict_jitted(
    w: np.ndarray,
    Ww: np.ndarray,
    CtC: np.ndarray,
    MC_col: np.ndarray,
    kidx: int,
    step: float,
    lam: float,
    n_inner: int,
) -> np.ndarray:
    """
    Strict FISTA inner loop recomputing gradient at each iterate (jitted).
    - w: initial vector (p,)
    - Ww: current whole Ww matrix (p x K)
    - CtC: Cw^T Cw (K x K)
    - MC_col: M @ Cw column (p,)
    - kidx: column index being updated
    - step: step size 1/L
    - lam: l1 penalty for this column
    - n_inner: inner iterations
    """
    p = w.shape[0]
    y = w.copy()
    t = 1.0
    for _ in range(n_inner):
        # compute gradient at y for column kidx
        grad = np.empty(p, dtype=w.dtype)
        for i in range(p):
            s = 0.0
            for j in range(CtC.shape[0]):
                val = Ww[i, j]
                if j == kidx:
                    val = y[i]
                s += val * CtC[j, kidx]
            grad[i] = s - MC_col[i]
        x = y - step * grad
        x = _soft_threshold_jitted(x, lam * step)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        # momentum
        y = x + ((t - 1.0) / t_new) * (x - w)
        w = x
        t = t_new
    return w


@njit
def _update_Ww_strict_numba(
    M: np.ndarray,
    Ww: np.ndarray,
    Cw: np.ndarray,
    lambda_w: np.ndarray,
    n_inner: int,
) -> tuple:
    """
    One outer iteration updating all columns of Ww using strict FISTA inner loops.
    Returns (Ww_new, max_change)
    """
    p, K = Ww.shape
    # compute CtC = Cw^T Cw
    CtC = np.empty((K, K), dtype=M.dtype)
    for i in range(K):
        for j in range(K):
            s = 0.0
            for r in range(Cw.shape[0]):
                s += Cw[r, i] * Cw[r, j]
            CtC[i, j] = s
    # precompute MC = M @ Cw (p x K)
    MC = np.empty((p, K), dtype=M.dtype)
    for i in range(p):
        for k in range(K):
            s = 0.0
            for r in range(Cw.shape[0]):
                s += M[i, r] * Cw[r, k]
            MC[i, k] = s
    # Lipschitz bound: use row-sum of CtC as approximation
    max_row = 0.0
    for i in range(K):
        row_sum = 0.0
        for j in range(K):
            row_sum += abs(CtC[i, j])
        if row_sum > max_row:
            max_row = row_sum
    L = 2.0 * max_row + 1e-12
    step = 1.0 / L
    Ww_new = Ww.copy()
    maxchg = 0.0
    for k in range(K):
        MC_col = MC[:, k]
        wk = Ww[:, k].copy()
        w_updated = _fista_inner_strict_jitted(
            wk,
            Ww,
            CtC,
            MC_col,
            k,
            step,
            float(lambda_w[k]),
            n_inner,
        )
        # normalize column
        nrm = 0.0
        for i in range(p):
            nrm += w_updated[i] * w_updated[i]
        nrm = np.sqrt(nrm)
        if nrm > 1e-12:
            for i in range(p):
                w_updated[i] = w_updated[i] / nrm
        # store and measure change
        for i in range(p):
            diff = abs(w_updated[i] - Ww[i, k])
            if diff > maxchg:
                maxchg = diff
            Ww_new[i, k] = w_updated[i]
    return Ww_new, maxchg


# ----------------------------
# RMT utilities (MP fit and cleaning)
# ----------------------------


def _mp_density(xs: np.ndarray, q: float, sigma2: float) -> np.ndarray:
    lam_minus = sigma2 * max(0.0, (1.0 - np.sqrt(q)) ** 2)
    lam_plus = sigma2 * (1.0 + np.sqrt(q)) ** 2
    dens = np.zeros_like(xs)
    mask = (xs > lam_minus) & (xs < lam_plus)
    if np.any(mask):
        x = xs[mask]
        numer = np.sqrt((lam_plus - x) * (x - lam_minus))
        denom = 2.0 * np.pi * q * sigma2 * x
        dens[mask] = numer / denom
    area = np.trapz(dens, xs)
    if area > 0:
        dens /= area
    return dens


def estimate_sigma2_mp_golden(
    eigs: np.ndarray,
    n_samples: int,
    tol: float = 1e-6,
    maxiter: int = 40,
) -> float:
    p = eigs.size
    low = max(1e-12, np.percentile(eigs, 1) * 0.05)
    high = max(1e-12, np.percentile(eigs, 99) * 2.0)
    gr = (np.sqrt(5) - 1) / 2.0
    a, b = low, high

    def loss(sigma: float) -> float:
        bins = np.linspace(np.min(eigs), np.max(eigs), 120)
        emp_hist, _ = np.histogram(eigs, bins=bins, density=True)
        centers = 0.5 * (bins[:-1] + bins[1:])
        dens = _mp_density(centers, float(p) / float(max(1, n_samples)), sigma)
        return float(np.mean((emp_hist - dens) ** 2))

    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = loss(c)
    fd = loss(d)
    it = 0
    while (b - a) > tol and it < maxiter:
        if fc < fd:
            b, fd = d, fc
            d = c
            c = b - gr * (b - a)
            fc = loss(c)
        else:
            a, fc = c, fd
            c = d
            d = a + gr * (b - a)
            fd = loss(d)
        it += 1
    return float(0.5 * (a + b))


def _rmt_clean_eigenvalues(
    eigvals: np.ndarray,
    n_samples: int,
    method: str = "clip_to_sigma2",
    sigma2_user=None,
) -> np.ndarray:
    p = eigvals.size
    q = float(p) / float(max(1, n_samples))
    if sigma2_user is not None:
        sigma2 = float(sigma2_user)
    else:
        try:
            sigma2 = estimate_sigma2_mp_golden(eigvals, n_samples)
        except Exception:
            edge_plus = (1.0 + np.sqrt(q)) ** 2
            edge_minus = max(0.0, (1.0 - np.sqrt(q)) ** 2)
            denom = 0.5 * (edge_plus + edge_minus)
            median_eig = float(np.median(eigvals))
            sigma2 = median_eig / max(denom, 1e-12)
    lambda_plus = sigma2 * (1.0 + np.sqrt(q)) ** 2
    lambda_minus = sigma2 * max(0.0, (1.0 - np.sqrt(q)) ** 2)
    if method == "clip_to_sigma2":
        cleaned = np.where(eigvals > lambda_plus, eigvals, sigma2)
    elif method == "clip_to_lambda_plus":
        cleaned = np.where(eigvals > lambda_plus, eigvals, lambda_plus)
    elif method == "soft_shrink":
        scale = 0.5
        cleaned = eigvals.copy()
        mask = eigvals < lambda_plus
        cleaned[mask] = sigma2 + (eigvals[mask] - lambda_minus) * scale
        cleaned = np.maximum(cleaned, sigma2)
    else:
        raise ValueError("Unknown RMT method: " + str(method))
    return cleaned


# ----------------------------
# Procrustes helper
# ----------------------------


def orthogonal_procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(B.T @ A)
    return U @ Vt


# ----------------------------
# Config dataclass
# ----------------------------


@dataclass
class PLS2Config:
    n_components: int = 4
    shrink_x: float = 0.2
    shrink_xy: float = 0.0
    ridge_alpha: float = 1e-4
    sigma_xx_reg: float = 1e-8
    group_map = None
    lambda_w = 0.0
    sparse_solver = "fista"
    sparse_maxiter: int = 200
    sparse_tol: float = 1e-6
    sparse_inner: int = 10
    use_rmt_xx: bool = False
    use_rmt_m: bool = False
    rmt_method: str = "clip_to_sigma2"
    rmt_sigma2 = None


# ----------------------------
# PLS2 estimator
# ----------------------------


class PLS2(BaseEstimator, RegressorMixin):
    """
    PLS2 estimator with whitening, optional RMT cleaning and numba-accelerated
    strict-FISTA sparse solver.

    Public attributes after fit:
      W_ : ndarray (p x K)
      coef_ : ndarray (p x q)
      intercept_ : ndarray (q,)
    """

    def __init__(self, cfg: PLS2Config | None = None):
        self.cfg = cfg if cfg is not None else PLS2Config()

    def get_params(self, deep: bool = True):
        params = {"cfg": self.cfg}
        for k, v in self.cfg.__dict__.items():
            params[f"cfg__{k}"] = v
        return params

    def set_params(self, **params):
        if "cfg" in params:
            self.cfg = params.pop("cfg")
        to_update = {}
        for key in list(params.keys()):
            if key.startswith("cfg__"):
                fld = key.split("__", 1)[1]
                to_update[fld] = params.pop(key)
        for f, val in to_update.items():
            if hasattr(self.cfg, f):
                setattr(self.cfg, f, val)
            else:
                raise ValueError("Unknown cfg field: " + str(f))
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _compute_sxx_clean(self, Xc: np.ndarray) -> np.ndarray:
        n = Xc.shape[0]
        sxx = (Xc.T @ Xc) / float(n)
        if getattr(self.cfg, "group_map") is None:
            diag = np.diag(np.diag(sxx))
            sxx_clean = self.cfg.shrink_x * diag + (1.0 - self.cfg.shrink_x) * sxx
        else:
            sxx_clean = sxx.copy()
            groups = np.unique(self.cfg.group_map)
            for g in groups:
                idx = np.where(self.cfg.group_map == g)[0]
                sub = sxx[np.ix_(idx, idx)]
                diag_sub = np.diag(np.diag(sub))
                sxx_clean[np.ix_(idx, idx)] = (
                    self.cfg.shrink_x * diag_sub + (1.0 - self.cfg.shrink_x) * sub
                )
        sxx_clean = 0.5 * (sxx_clean + sxx_clean.T)
        if getattr(self.cfg, "use_rmt_xx"):
            evals, evecs = np.linalg.eigh(sxx_clean)
            evals_clean = _rmt_clean_eigenvalues(
                evals, Xc.shape[0], method=self.cfg.rmt_method, sigma2_user=self.cfg.rmt_sigma2
            )
            sxx_clean = evecs @ np.diag(evals_clean) @ evecs.T
            sxx_clean = 0.5 * (sxx_clean + sxx_clean.T)
        return sxx_clean

    def _compute_whitened_M(self, sxx_clean: np.ndarray, sxy: np.ndarray):
        evals, evecs = np.linalg.eigh(sxx_clean)
        evals_safe = np.maximum(evals, self.cfg.sigma_xx_reg)
        inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals_safe)) @ evecs.T
        M = inv_sqrt @ sxy
        if getattr(self.cfg, "use_rmt_m"):
            warnings.warn(
                "use_rmt_m=True is heuristic; validate carefully on OOS folds.", UserWarning
            )
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            eigs_mm = S ** 2
            evals_mm_clean = _rmt_clean_eigenvalues(
                eigs_mm, sxx_clean.shape[0], method=self.cfg.rmt_method, sigma2_user=self.cfg.rmt_sigma2
            )
            S_clean = np.sqrt(np.maximum(evals_mm_clean, 0.0))
            M = U @ np.diag(S_clean) @ Vt
            return M, inv_sqrt, (U, S_clean, Vt)
        return M, inv_sqrt, None

    def _sparse_alternating(self, M: np.ndarray, K: int, lambda_w: np.ndarray) -> tuple:
        p, q = M.shape
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        K_eff = min(K, U.shape[1])
        Ww = U[:, :K_eff].copy()
        Cw = Vt[:K_eff, :].T.copy()
        for iteration in range(self.cfg.sparse_maxiter):
            CtC = Cw.T @ Cw
            # approximate spectral norm: use 2*max row sum for safety
            L = 2.0 * np.max(np.sum(np.abs(CtC), axis=1)) + 1e-12
            step = 1.0 / L
            Ww_new, maxchg = _update_Ww_strict_numba(
                M, Ww, Cw, lambda_w.astype(np.float64), self.cfg.sparse_inner
            )
            WtW = Ww_new.T @ Ww_new
            WtW_inv = np.linalg.inv(WtW + 1e-12 * np.eye(WtW.shape[0]))
            Cw = (M.T @ Ww_new) @ WtW_inv
            Ww = Ww_new
            if maxchg < self.cfg.sparse_tol:
                break
        return Ww, Cw

    def fit(self, X: np.ndarray, Y: np.ndarray):
        X = check_array(X, ensure_2d=True, dtype=float)
        Y = check_array(Y, ensure_2d=True, dtype=float)
        n, p = X.shape
        self.X_mean_ = X.mean(axis=0)
        self.Y_mean_ = Y.mean(axis=0)
        Xc = X - self.X_mean_
        Yc = Y - self.Y_mean_
        sxx_raw = (Xc.T @ Xc) / float(n)
        sxy_raw = (Xc.T @ Yc) / float(n)
        sxx_clean = self._compute_sxx_clean(Xc)
        sxy = (1.0 - self.cfg.shrink_xy) * sxy_raw
        M, inv_sqrt, m_svd_info = self._compute_whitened_M(sxx_clean, sxy)
        K = int(self.cfg.n_components)
        lambda_w = self.cfg.lambda_w
        if np.isscalar(lambda_w):
            lambda_w = float(lambda_w)
            lambda_w = np.array([lambda_w] * K, dtype=float)
        else:
            lambda_w = np.asarray(lambda_w, dtype=float)
            if lambda_w.size != K:
                raise ValueError("lambda_w must be scalar or of length n_components")
        if np.any(lambda_w > 0.0):
            Ww, Cw = self._sparse_alternating(M, K, lambda_w)
            W = inv_sqrt @ Ww
            C = Cw
        else:
            U, Svals, Vt = np.linalg.svd(M, full_matrices=False)
            U_k = U[:, :K]
            Vt_k = Vt[:K, :]
            W = inv_sqrt @ U_k
            C = Vt_k.T
        norms = np.sqrt(np.sum(W * (sxx_clean @ W), axis=0))
        norms = np.where(norms == 0.0, 1.0, norms)
        W = W / norms
        T_scores = Xc @ W
        ridge = Ridge(alpha=self.cfg.ridge_alpha)
        ridge.fit(T_scores, Yc)
        coef = getattr(ridge, "coef_", None)
        if coef is None:
            raise RuntimeError("Ridge did not produce coefficients")
        coef = np.atleast_2d(coef)
        if coef.shape[1] != W.shape[1]:
            coef = coef.T
        B = W @ coef.T
        self.W_ = W
        self.C_ = C
        self.T_scores_ = T_scores
        self.ridge_ = ridge
        self.coef_ = B
        self.intercept_ = (self.Y_mean_ - (self.X_mean_ @ B)).reshape(-1)
        resid = Yc - ridge.predict(T_scores)
        self.sigma2_ = np.maximum(np.mean(resid ** 2, axis=0), 1e-12)
        self.sxx_clean_ = sxx_clean
        self.sxx_raw_ = sxx_raw
        self.sxy_raw_ = sxy_raw
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["W_", "coef_", "intercept_"])
        X = check_array(X, ensure_2d=True, dtype=float)
        return X @ self.coef_ + self.intercept_

    def predict_interval(self, X: np.ndarray, z: float = 1.96):
        check_is_fitted(self, ["W_", "ridge_", "T_scores_", "sigma2_"])
        X = check_array(X, ensure_2d=True, dtype=float)
        s_new = (X - self.X_mean_) @ self.W_
        tt = self.T_scores_.T @ self.T_scores_
        cov_base = np.linalg.inv(tt + self.cfg.ridge_alpha * np.eye(self.W_.shape[1]))
        preds = self.predict(X)
        n_obs, q = preds.shape
        stds = np.empty((n_obs, q), dtype=float)
        for j in range(q):
            sigma2_j = float(self.sigma2_[j])
            cov_j = sigma2_j * cov_base
            tmp = s_new @ cov_j
            var_vec = np.sum(tmp * s_new, axis=1)
            stds[:, j] = np.sqrt(np.maximum(var_vec, 0.0))
        lower = preds - z * stds
        upper = preds + z * stds
        return preds, lower, upper


# ----------------------------
# Feature / label utilities
# ----------------------------


def build_intraday_label(
    mid: pd.Series,
    horizon_hours: float = 6.0,
) -> tuple:
    """
    Build intraday-only forward log-return label. Exclude timestamps where the
    forward horizon crosses session boundary.

    Parameters
    ----------
    mid : pd.Series
        Midprice indexed by pd.DatetimeIndex (timezone-aware recommended).
    horizon_hours : float
        Prediction horizon in hours (e.g., 6.0).

    Returns
    -------
    y : pd.Series
        Forward intraday log return (NaN where invalid).
    valid_mask : pd.Series
        Boolean mask True where label is intraday-valid.
    """
    if not isinstance(mid.index, pd.DatetimeIndex):
        raise ValueError("mid must be a pandas Series indexed by a DatetimeIndex")
    dt = pd.Timedelta(hours=horizon_hours)
    forward_time = mid.index + dt
    # get next available observation at or after forward_time
    idx = mid.index.searchsorted(forward_time, side="left")
    invalid_idx = idx >= len(mid)
    # temporary assignment to avoid indexing errors
    idx[invalid_idx] = len(mid) - 1
    future_prices = mid.iloc[idx].to_numpy()
    # same session check (normalize handles DST, holidays by date)
    same_session = (mid.index.normalize() == mid.index[idx].normalize())
    valid = (~invalid_idx) & same_session
    ret = np.full(len(mid), np.nan, dtype=float)
    arr_mid = mid.to_numpy(dtype=float)
    ret[valid] = np.log(future_prices[valid]) - np.log(arr_mid[valid])
    return pd.Series(ret, index=mid.index), pd.Series(valid, index=mid.index)


def make_vol_normalized_targets(
    price: np.ndarray,
    horizon: int,
    vol_lookback: int,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Build volatility-normalized forward returns y_t = r_{t→t+h} / sigma_t.

    Parameters
    ----------
    price : ndarray (T,)
    horizon : int (steps)
    vol_lookback : int (steps for realized vol)
    eps : float floor for sigma

    Returns
    -------
    y : ndarray (T - horizon,)
    """
    price = np.asarray(price, dtype=float)
    if price.ndim != 1:
        raise ValueError("price must be 1D array")
    n = price.size
    if n <= horizon + vol_lookback:
        raise ValueError("not enough data for horizon + vol_lookback")
    # forward simple return
    fwd = (price[horizon:] - price[:-horizon]) / price[:-horizon]
    # realized vol from past returns
    rets = np.empty(n - 1, dtype=float)
    rets[:] = (price[1:] - price[:-1]) / price[:-1]
    sigma = np.empty(n - horizon, dtype=float)
    sigma[:] = np.nan
    # sigma aligned at time t uses returns up to t (exclusive)
    for t in range(vol_lookback, n - horizon):
        window = rets[t - vol_lookback:t]
        sigma[t] = np.std(window, ddof=0)
    sigma = np.maximum(sigma, eps)
    # remove initial entries where sigma is nan
    valid = ~np.isnan(sigma)
    y = np.full_like(fwd, np.nan, dtype=float)
    y[valid] = fwd[valid] / sigma[valid]
    return y


# ----------------------------
# Purged + Embargoed CV splitter
# ----------------------------


class PurgedWalkForwardSplit:
    """
    Purged walk-forward splitter with embargo.
    Yields (train_idx, test_idx) on integer index basis.

    Parameters
    ----------
    n_splits : int
    test_size : int | None
        If None, uses floor(n / (n_splits + 1)).
    embargo : int (timesteps)
    """

    def __init__(self, n_splits: int = 5, test_size=None, embargo: int = 0):
        self.n_splits = int(n_splits)
        self.test_size = test_size
        self.embargo = int(embargo)

    def split(self, X: np.ndarray, y: np.ndarray | None = None):
        n = X.shape[0]
        if self.test_size is None:
            test_size = max(1, n // (self.n_splits + 1))
        else:
            test_size = int(self.test_size)
        starts = range(n - self.n_splits * test_size, n, test_size)
        for start in starts:
            train_end = start
            test_start = start
            test_end = min(n, start + test_size)
            embargo_start = max(0, test_start - self.embargo)
            if embargo_start <= 0:
                train_idx = np.arange(0, train_end)
            else:
                train_idx = np.arange(0, embargo_start)
            test_idx = np.arange(test_start, test_end)
            yield train_idx, test_idx


# ----------------------------
# Ridge tuning helper (purged CV)
# ----------------------------


def tune_ridge_via_cv(
    pipeline: Pipeline,
    X: np.ndarray,
    Y: np.ndarray,
    alphas: list | np.ndarray,
    cv: PurgedWalkForwardSplit | None = None,
    scoring: str = "neg_mean_squared_error",
    n_jobs: int = 1,
):
    """
    Tune ridge_alpha (cfg.ridge_alpha) using GridSearchCV over purged walk-forward CV.
    The pipeline should be created via make_hybrid_pipeline(cfg, Z).
    """
    from sklearn.model_selection import GridSearchCV

    if cv is None:
        cv = PurgedWalkForwardSplit(n_splits=5, embargo=5)
    param_grid = {"model__regressor__cfg__ridge_alpha": list(alphas)}
    gs = GridSearchCV(
        pipeline, param_grid, cv=cv.split(X, Y), scoring=scoring, n_jobs=n_jobs
    )
    gs.fit(X, Y)
    return gs


# ----------------------------
# Lightweight pipeline builders
# ----------------------------


class YOrth(BaseEstimator):
    def __init__(self, Z: pd.DataFrame | None = None):
        self.Z = Z

    def fit(self, y: np.ndarray, X=None):
        if self.Z is None:
            self.gamma_ = None
            return self
        Z = check_array(self.Z, ensure_2d=True, dtype=float)
        y = check_array(y, ensure_2d=True, dtype=float)
        zz = Z.T @ Z
        self.gamma_ = np.linalg.pinv(zz) @ (Z.T @ y)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = check_array(y, ensure_2d=True, dtype=float)
        if getattr(self, "gamma_", None) is None:
            return y
        return y - self.Z @ self.gamma_


def make_feature_pipeline(n_quantiles: int = 1000) -> Pipeline:
    return Pipeline(
        [
            ("qt", QuantileTransformer(output_distribution="normal", n_quantiles=n_quantiles, subsample=100000)),
            ("scale", StandardScaler()),
        ]
    )


def make_target_transformer(Z: pd.DataFrame | None = None) -> Pipeline:
    return Pipeline([("orth", YOrth(Z=Z)), ("scale", StandardScaler())])


def make_hybrid_pipeline(cfg: PLS2Config | None = None, Z: pd.DataFrame | None = None) -> Pipeline:
    cfg = cfg if cfg is not None else PLS2Config()
    pls = PLS2(cfg=cfg)
    model = TransformedTargetRegressor(regressor=pls, transformer=make_target_transformer(Z=Z))
    return Pipeline([("xpipe", make_feature_pipeline()), ("model", model)])


# ----------------------------
# Demo: end-to-end synthetic intraday example
# ----------------------------


def demo_end_to_end(seed: int = 0) -> dict:
    """
    Demo:
     - build synthetic intraday-like prices with timestamps (1-minute cadence)
     - compute intraday-only 6h labels (masking those that cross session)
     - compute vol-normalized target
     - create simple features
     - fit PLS2 and print diagnostics
    """
    rng = np.random.default_rng(seed)
    # parameters
    minutes_per_day = 6 * 60  # reduced trading day for demo
    n_days = 10
    cadence = "1min"
    # build timestamps (naive trading days back-to-back)
    start = pd.Timestamp("2024-01-02 09:00")
    periods = n_days * minutes_per_day
    idx = pd.date_range(start=start, periods=periods, freq=cadence)
    # synthetic mid prices: small drift + day seasonality + noise
    t = np.arange(periods)
    seasonal = 0.1 * np.sin(2 * np.pi * (t % minutes_per_day) / minutes_per_day)
    prices = 100.0 + 0.001 * t + seasonal + 0.2 * rng.standard_normal(size=periods)
    mid = pd.Series(prices, index=idx)
    # build intraday labels (6 hours)
    y_log, valid = build_intraday_label(mid, horizon_hours=6.0)
    # drop invalid rows
    mid_valid = mid[valid]
    y_valid = y_log[valid].to_numpy(dtype=float)
    # vol-normalised target: use price array aligned with mid_valid
    price_valid = mid_valid.to_numpy(dtype=float)
    # horizon in steps: 6 hours / 1 minute
    horizon_steps = int(6 * 60 / 1)
    vol_lookback = int(60)  # 60 minutes realized vol
    # compute vol-normalized target (returns array length price_valid.size - horizon_steps)
    y_vol = make_vol_normalized_targets(price_valid, horizon_steps, vol_lookback)
    # align feature matrix: simple lag features
    p = price_valid.size
    L1 = 1
    L5 = 5
    L15 = 15
    # build lagged returns features (drop initial rows to keep shapes)
    ret = np.empty(p, dtype=float)
    ret[1:] = np.log(price_valid[1:]) - np.log(price_valid[:-1])
    ret[0] = 0.0
    X_full = np.column_stack(
        [
            np.roll(ret, L1),
            np.roll(ret, L5),
            np.roll(ret, L15),
            np.linspace(-1.0, 1.0, p).reshape(-1, 1)[:, 0:1].ravel(),  # simple linear time feature
        ]
    )
    # trim initial rows so that y_vol and X align
    keep = np.arange(len(y_vol))
    X = X_full[: len(y_vol), :3]  # pick first 3 columns
    Y = y_vol.reshape(-1, 1)
    # drop NaNs
    mask = ~np.isnan(Y).ravel()
    X = X[mask]
    Y = Y[mask].reshape(-1, 1)
    print("Shapes:", X.shape, Y.shape)
    # fit PLS2
    cfg = PLS2Config(n_components=3, shrink_x=0.15, use_rmt_xx=True, lambda_w=0.0)
    pipeline = make_hybrid_pipeline(cfg)
    pipeline.fit(X, Y)
    yhat = pipeline.predict(X)
    # simple in-sample IC
    ic = np.corrcoef(Y.ravel(), yhat.ravel())[0, 1]
    print("In-sample IC:", float(ic))
    # purged walk-forward CV
    splitter = PurgedWalkForwardSplit(n_splits=4, embargo=int(30))
    ics = []
    for tr, te in splitter.split(X):
        if len(tr) < 50:
            continue
        model = PLS2(cfg=cfg)
        model.fit(X[tr], Y[tr])
        yh = model.predict(X[te])
        if np.std(yh) == 0 or np.std(Y[te]) == 0:
            ics.append(0.0)
        else:
            ics.append(float(np.corrcoef(Y[te].ravel(), yh.ravel())[0, 1]))
    print("Purged CV mean IC:", float(np.mean(ics)) if len(ics) > 0 else None)
    return {
        "pipeline": pipeline,
        "cfg": cfg,
        "X": X,
        "Y": Y,
        "yhat": yhat,
        "ic_in": ic,
        "ic_cv": ics,
    }


if __name__ == "__main__":
    out = demo_end_to_end()
    print("Demo finished.")