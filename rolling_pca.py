import numpy as np
import pandas as pd


def halflife_to_decay(halflife):
    if halflife <= 0.:
        raise ValueError('halflife must be a positive real number')
    return np.exp(np.log(0.5) / halflife)


def exponential_decaying_weights(n: int, halflife: float=21):
    phi = halflife_to_decay(halflife)
    idx = np.arange(0, n, 1)[::-1]
    return np.power(phi, idx)


def pre_process_data(data, standardize=True):
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    res = data - data_mean
    if standardize:
        res /= data_std
    return res, data_mean, data_std


def pc_sign_flip(u, v ,v_prev=None, thresh_pin_pc1=0.0, thres_pin_pc=0.8):
    if v_prev is not None:
        # PC1 has a special threshold
        thresholds = np.hstack((thresh_pin_pc1, [thres_pin_pc] * (v.shape[0] - 1)))
        # Count number of sign flips
        sign_flip_number = np.sum(np.sign(v * v_prev) < 0, axis=1)
        # Array of booleans
        is_sign_flipped = sign_flip_number > thresholds * v.shape[0]
        # Flips columns / rows 
        v[is_sign_flipped, :] *= -1
        u[:, is_sign_flipped] *= -1
    return u, v


def rolling_pca(data, n=3, window=252, burn_in=120, standardize=True,
                thresh_pin_pc1=0.0, thres_pin_pc=0.2):
    """
    ftp://ftp.cea.fr/pub/unati/people/educhesnay/centralesupelec/StatisticsMachineLearningPythonDraft.pdf
    """
    loadings_dict = {}
    variances_dict = {}
    pcs_dict = {}
    data_mean_dict = {}
    data_std_dict = {}
    
    headers = ['PC{}'.format(i + 1) for i in range(data.shape[1])]    
    
    v_prev = None
    for i in tqdm(range(burn_in, len(data))):
        idx_start = max(0, i - window)
        _data = data.iloc[idx_start: i]
        if i > window:
            assert len(_data) == window, 'Not enough data'
        input_data, data_mean, data_std = pre_process_data(_data, standardize=standardize)
        u, s, v = np.linalg.svd(input_data, full_matrices=False)
        
        # Flip eigenvector signs if needed
        u, v = pc_sign_flip(u, v, v_prev)
        
        # Loadings = principal component directions = eigenvector of X'X 
        loadings = pd.DataFrame(v.copy().transpose(), columns=headers, index=_data.columns)
        variances = pd.Series((s ** 2) / len(_data), index=headers)
        # PCs are obtained by projection X_centered onto the PC directions
        # coordinates of the variables in the new orthogonal basis defined by V
        pcs = pd.DataFrame(u * s, index=_data.index, columns=headers)
        
        current_date = data.index[i]
        loadings_dict[current_date] = loadings
        variances_dict[current_date] = variances
        pcs_dict[current_date] = pcs.iloc[-1, ]
        data_mean_dict[current_date] = data_mean
        data_std_dict[current_date] = data_std
    
        v_prev = v
        
    results = {
        'loadings_panel': pd.Panel(loadings_dict),
        'pcs': pd.DataFrame(pcs_dict).T,
        'variance_explained': pd.DataFrame(variances_dict).T,
        'data_mean': pd.DataFrame(data_mean_dict).T,
        'data_std': pd.DataFrame(data_std_dict).T,
    }
    
    return results



