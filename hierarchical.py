import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.covariance import LedoitWolf
from scipy.cluster.hierarchy import linkage, fcluster
from typing import Optional, Tuple, List


# ------------------------
# Utilities: stationary bootstrap indices & decorrelation estimate
# ------------------------
def estimate_decorrelation_time(returns: pd.DataFrame, max_lag: int = 120) -> int:
    """
    Estimate a conservative median integrated autocorrelation time across assets.
    Returns an integer mean block length (conservative).
    """
    def tau_series(x: np.ndarray) -> float:
        x = x - np.nanmean(x)
        denom = np.nansum(x * x)
        if denom == 0.0:
            return 1.0
        rhos = []
        T = len(x)
        L = min(max_lag, max(10, T // 10))
        for k in range(1, L):
            acov = np.nansum(x[:-k] * x[k:])
            rho = acov / denom
            rhos.append(rho)
            if abs(rho) < 0.01:
                break
        return max(1.0, 1.0 + 2.0 * float(np.sum(rhos)))

    taus: List[float] = []
    for col in returns.columns:
        arr = returns[col].to_numpy(dtype=float, copy=False)
        if np.all(np.isnan(arr)):
            taus.append(1.0)
        else:
            taus.append(tau_series(arr))
    med = float(np.nanmedian(taus))
    T = len(returns)
    # conservative scaling and clipping
    L = int(max(10, min(T // 2, int(np.ceil(1.5 * med)))))
    return L


def stationary_bootstrap_indices(T: int, mean_block: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Politis-Romano stationary bootstrap indices length T, mean block ~ mean_block.
    """
    if rng is None:
        rng = default_rng()
    p = 1.0 / float(max(1, mean_block))
    idx = np.empty(T, dtype=int)
    i = 0
    while i < T:
        start = int(rng.integers(0, T))
        run_len = int(rng.geometric(p))
        run_len = min(run_len, T - i)
        block = (np.arange(start, start + run_len) % T).astype(int)
        idx[i : i + run_len] = block
        i += run_len
    return idx


# ------------------------
# Covariance shrinkage and RMT utilities
# ------------------------
def shrinkage_corr(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Ledoit-Wolf shrinkage covariance -> correlation matrix.
    Assumes returns is T x N and has no structural NaNs (complete cross-section).
    """
    X = returns.values
    lw = LedoitWolf().fit(X)
    cov = lw.covariance_
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    corr = np.clip(corr, -1.0, 1.0)
    return pd.DataFrame(corr, index=returns.columns, columns=returns.columns)


def rmt_eigvals_and_lambda_plus(corr: pd.DataFrame, T: int) -> Tuple[np.ndarray, float]:
    """
    Return eigenvalues sorted descending and the MP upper bound lambda_plus.
    """
    N = corr.shape[0]
    eigvals = np.linalg.eigvalsh(corr.values)
    eigvals = np.sort(eigvals)[::-1]  # descending
    q = float(N) / float(max(1, T))
    # Marchenko-Pastur upper bound for correlation matrix with unit variance assumption
    if q <= 1.0:
        lambda_plus = (1.0 + np.sqrt(q)) ** 2
    else:
        # when N > T, the relevant form uses 1/q
        lambda_plus = (1.0 + np.sqrt(1.0 / q)) ** 2
    return eigvals, float(lambda_plus)


def rmt_denoise_corr(corr: pd.DataFrame, T: int) -> pd.DataFrame:
    """
    Denoise correlation matrix by replacing eigenvalues in the MP bulk by their average.
    """
    eigvals, eigvecs = np.linalg.eigh(corr.values)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    N = corr.shape[0]
    q = float(N) / float(max(1, T))
    if q <= 1.0:
        lambda_min = (1.0 - np.sqrt(q)) ** 2
        lambda_max = (1.0 + np.sqrt(q)) ** 2
    else:
        lambda_min = (1.0 - np.sqrt(1.0 / q)) ** 2
        lambda_max = (1.0 + np.sqrt(1.0 / q)) ** 2

    noisy_mask = (eigvals >= lambda_min) & (eigvals <= lambda_max)
    if noisy_mask.any():
        avg_noise = float(np.mean(eigvals[noisy_mask]))
    else:
        avg_noise = 0.0
    eigvals_clean = np.where(noisy_mask, avg_noise, eigvals)
    C_clean = eigvecs @ np.diag(eigvals_clean) @ eigvecs.T
    C_clean = np.clip(C_clean, -1.0, 1.0)
    return pd.DataFrame(C_clean, index=corr.index, columns=corr.columns)


# ------------------------
# Clustering helpers
# ------------------------
def corr_to_dist(corr: pd.DataFrame) -> np.ndarray:
    """
    Convert correlation to Euclidean distance matrix (NxN).
    d_ij = sqrt(0.5*(1 - rho_ij))
    """
    D = np.sqrt(0.5 * (1.0 - corr.values))
    # numerical safety
    D = np.where(np.isfinite(D), D, 0.0)
    return D


def linkage_from_dist_matrix(dist: np.ndarray, method: str = "ward"):
    """
    Given full NxN distance matrix, return linkage matrix (scipy) using condensed form.
    """
    triu = dist[np.triu_indices_from(dist, k=1)]
    Z = linkage(triu, method=method)
    return Z


def cluster_labels_from_linkage(Z, k: int) -> np.ndarray:
    """
    Cut linkage at k clusters -> integer labels (1..k), length N.
    """
    labs = fcluster(Z, k, criterion="maxclust")
    return labs


# ------------------------
# Stability / co-association for one candidate k
# ------------------------
def coassociation_for_k(
    returns: pd.DataFrame,
    k: int,
    B: int = 500,
    mean_block: Optional[int] = None,
    rng_seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    For fixed number of clusters k, perform B stationary bootstraps and
    return co-association matrix C (NxN) and an array labels_b of shape (B, N)
    containing cluster labels for each bootstrap.
    """
    T, N = returns.shape
    names = returns.columns.tolist()
    rng = default_rng(rng_seed)
    if mean_block is None:
        mean_block = estimate_decorrelation_time(returns)
        mean_block = int(max(20, min(T // 2, mean_block)))

    # storage
    C = np.zeros((N, N), dtype=float)
    labels_b = np.zeros((B, N), dtype=int)

    for b in range(B):
        idx = stationary_bootstrap_indices(T, mean_block, rng)
        sample = returns.iloc[idx]

        # cleaning pipeline
        corr = shrinkage_corr(sample)
        corr = rmt_denoise_corr(corr, len(sample))
        dist = corr_to_dist(corr)
        Z = linkage_from_dist_matrix(dist, method="ward")

        labs = cluster_labels_from_linkage(Z, k)
        labels_b[b, :] = labs

        # update co-association matrix
        for i in range(N):
            for j in range(i + 1, N):
                if labs[i] == labs[j]:
                    C[i, j] += 1.0
                    C[j, i] += 1.0

    C /= float(B)
    np.fill_diagonal(C, 1.0)
    C_df = pd.DataFrame(C, index=names, columns=names)
    return C_df, labels_b


# ------------------------
# Score function Q(k): average in-cluster co-association using consensus labels
# ------------------------
def stability_score_from_coassoc(C: pd.DataFrame, consensus_labels: np.ndarray) -> float:
    """
    Given co-association matrix C (NxN) and consensus integer labels (length N),
    compute average in-cluster co-association Q:
      Q = mean_{i<j, same cluster} C_ij
    """
    arr = C.values
    N = arr.shape[0]
    mask = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(i + 1, N):
            if consensus_labels[i] == consensus_labels[j]:
                mask[i, j] = True
                mask[j, i] = True
    # collect upper-triangular pairs
    if not mask.any():
        return 0.0
    selected = arr[mask]
    return float(np.mean(selected))


# ------------------------
# Main auto-tune function
# ------------------------
def auto_tune_clusters(
    returns: pd.DataFrame,
    B: int = 500,
    mean_block: Optional[int] = None,
    rng_seed: Optional[int] = 0,
    k_extra: int = 2,
    k_min: int = 2,
    k_max_override: Optional[int] = None,
) -> Tuple[int, pd.Series, pd.DataFrame]:
    """
    Auto-tune number of clusters and return:
      - best_k: selected number of clusters
      - final_labels: pd.Series (index=asset names, values = cluster id)
      - C_best: co-association matrix for best_k
    Procedure:
      1) compute eigenvalues + lambda_plus to get k_factors (number of eigvals > lambda_plus)
      2) define candidate k in [k_min, min(k_factors + k_extra, k_max_override or N-1)]
      3) for each candidate k, compute co-association matrix (B bootstraps)
         and consensus labels (per-asset majority vote across bootstraps)
      4) compute stability score Q(k) and pick k maximizing Q
    """
    T, N = returns.shape
    names = returns.columns.tolist()
    rng = default_rng(rng_seed)

    # 1) RMT factor count
    corr_full = shrinkage_corr(returns)
    eigvals, lambda_plus = rmt_eigvals_and_lambda_plus(corr_full, T)
    k_factors = int(np.sum(eigvals > lambda_plus))
    k_factors = max(1, k_factors)

    # candidate range
    if k_max_override is None:
        k_max = min(N - 1, k_factors + k_extra)
    else:
        k_max = min(N - 1, k_max_override)
    k_min = max(2, k_min)
    candidate_ks = list(range(k_min, max(k_min, k_max) + 1))

    if mean_block is None:
        mean_block = estimate_decorrelation_time(returns)
        mean_block = int(max(20, min(T // 2, mean_block)))

    best_k = None
    best_score = -np.inf
    best_C = None
    best_consensus = None

    # loop candidates
    for k in candidate_ks:
        C_k, labels_b = coassociation_for_k(
            returns=returns,
            k=k,
            B=B,
            mean_block=mean_block,
            rng_seed=int(rng.integers(0, 2 ** 31 - 1)),
        )

        # consensus labels: majority vote across bootstraps for each asset
        # labels_b shape (B, N)
        from scipy.stats import mode  # local import for clarity
        mode_res = mode(labels_b, axis=0, keepdims=False)
        consensus_labels = mode_res.mode.astype(int)

        # compute stability score
        Qk = stability_score_from_coassoc(C_k, consensus_labels)
        # prefer smaller k in tie by adding tiny penalty
        penalty = 1e-8 * k
        score = Qk - penalty

        if score > best_score:
            best_score = score
            best_k = k
            best_C = C_k.copy()
            best_consensus = consensus_labels.copy()

    # final labels as pd.Series with best_k cluster ids
    final_labels = pd.Series(best_consensus, index=names, name="cluster")
    return int(best_k), final_labels, best_C


# ------------------------
# Example usage:
# ------------------------
# Suppose `df` is your T x N DataFrame of daily returns (no NaNs).
# best_k, labels, C = auto_tune_clusters(df, B=600, mean_block=None, rng_seed=42)
# print("Auto-selected k =", best_k)
# print(labels.value_counts().sort_index())
# Plot co-association heatmap sorted by cluster if you want.