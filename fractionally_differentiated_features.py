import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def get_weights(d, size):
    w = [1]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    return np.array(w[::-1])


def frac_diff(series, d, thres=0.01):
    '''
    Increasing width window, with treatment of NaNs 
    Note 1: For thres=1, nothing is skipped. 
    Note 2: d can be any positive fractional, not necessarily bounded [0,1]. 
    '''
    # 1) Compute weights for the longest series
    w = get_weights(d, series.shape[0])
    # 2) Determine initial calcs to be skipped based on weight-loss threshold
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_ > thres].shape[0]
    # 3) Apply weights to values
    s = []
    for iloc in range(skip, series.shape[0]):
        s.append(np.dot(w[-(iloc+1):], series.iloc[:(iloc+1)]))
    s = pd.Series(s)
    s.name = series.name
    s.index = series.index[skip:]
    return s


def frac_diff_ffd(series, d, thres=1e-5):
    w = get_weights(d, len(series))
    w = w[np.abs(w) > thres]
    result = series.rolling(len(w)).apply(lambda x: np.dot(w, x)).dropna()
    return result


def get_diff_factor(series):
    d = 0.05
    while d != 1:
        pvalue = adfuller(frac_diff_ffd(series, d))[1]
        if pvalue > 0.05:
            break
        d += 0.05
    return d - 0.05
