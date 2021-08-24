from typing import Tuple
import numpy as np
import pandas as pd


def resample_bar(data, type, **kwargs):
    if type == 'standard':
        return _stardard_bar(data, subtype=kwargs['subtype'], size=kwargs['size'])
    elif type == 'infomation_driven':
        return _infomation_driven_bar(data, subtype=kwargs['subtype'], alpha=kwargs['alpha'], limit=kwargs['limit'], initial_t=kwargs['initial_t'])
    else:
        raise Exception(f'Unknown type {type}')


def _stardard_bar(data, subtype, size):
    abar = data.copy()
    if subtype == 'amount':
        abar['amount'] = abar[['open', 'high', 'low', 'close']].mean(
            axis=1) * abar['volume']
        abar['cum'] = abar.amount.cumsum()
    elif subtype == 'volume':
        abar['cum'] = abar.volume.cumsum()
    else:
        raise Exception(f'Unknown subtype {subtype}')
    abar['cum'] //= size
    aopen = abar.groupby('cum')['open'].first()
    ahigh = abar.groupby('cum')['high'].max()
    alow = abar.groupby('cum')['low'].min()
    aclose = abar.groupby('cum')['close'].last()
    avolume = abar.groupby('cum')['volume'].sum()
    atime = abar.reset_index().groupby('cum')['index'].apply(
        lambda x: (x.iloc[-1] - x.iloc[0]).seconds//60+1)
    atime.name = 'time'

    result = pd.concat([aopen, ahigh, alow, aclose, avolume,
                       atime], axis=1).reset_index(drop=True)
    return result


def _infomation_driven_bar(data, subtype: Tuple[str, str], alpha, limit, initial_t):
    ibar = data.copy()
    if subtype[1] == 'amount':
        ibar['amount'] = ibar[['open', 'high', 'low', 'close']].mean(
            axis=1) * ibar['volume']
    ibar['b'] = np.sign(ibar['close']-ibar['open']
                        ).replace(to_replace=0, method='ffill')
    ibar.loc[ibar['b'] == 1, 'b+'] = ibar.loc[ibar['b'] == 1, subtype[1]]
    ibar['b+'] = ibar['b+'].cumsum().ffill()
    ibar.loc[ibar['b'] == -1, 'b-'] = -ibar.loc[ibar['b'] == -1, subtype[1]]
    ibar['b-'] = ibar['b-'].cumsum().ffill()
    ibar['theta'] = (ibar['b'] * ibar[subtype[1]]).cumsum()
    ibar['T'] = 0

    t = initial_t
    b = ibar['b'].iloc[:t]
    vp = b[b == 1].sum() / t * ibar.iloc[:t].loc[b == 1, subtype[1]].mean()
    vm = b[b == -1].sum() / t * ibar.iloc[:t].loc[b == -1, subtype[1]].mean()
    counter, l = 0, t
    while l < ibar.shape[0]:
        print(f'estimated progress {l/ibar.shape[0]:.2%}', end='\r')
        if subtype[0] == 'imbalance':
            threshold = t * abs(vp + vm)
            pivot = (np.abs(ibar['b+']+ibar['b-']) >=
                     threshold).reset_index(drop=True)[l:]
        elif subtype[0] == 'run':
            threshold = t * max(vp, -vm)
            pivot = (np.maximum(ibar['b+'], -ibar['b-']) >=
                     threshold).reset_index(drop=True)[l:]
        if not pivot.any():
            r = pivot.index[-1]
        else:
            r = pivot[pivot == True].index[0]
        r = min(r, l+limit)
        ibar['T'].iloc[l:r+1] = counter
        b_ = ibar['b'].iloc[l:r+1]
        t_ = r - l + 1
        vp_ = b_[b_ == 1].sum() / t_ * ibar.iloc[l:r +
                                                 1].loc[b_ == 1, subtype[1]].mean()
        vp_ = 0 if np.isnan(vp_) else vp_
        vm_ = b_[b_ == -1].sum() / t_ * ibar.iloc[l:r +
                                                  1].loc[b_ == -1, subtype[1]].mean()
        vm_ = 0 if np.isnan(vm_) else vm_
        t, vp, vm = alpha*t_ + (1-alpha)*t, alpha*vp_ + \
            (1-alpha)*vp, alpha*vm_ + (1-alpha)*vm
        l = r + 1
        counter += 1
        ibar['b+'] -= ibar['b+'].iloc[r]
        ibar['b-'] -= ibar['b-'].iloc[r]

    iopen = ibar.groupby('T')['open'].first()
    ihigh = ibar.groupby('T')['high'].max()
    ilow = ibar.groupby('T')['low'].min()
    iclose = ibar.groupby('T')['close'].last()
    ivolume = ibar.groupby('T')['volume'].sum()
    itime = ibar.reset_index().groupby('T')['index'].apply(
        lambda x: (x.iloc[-1] - x.iloc[0]).seconds//60+1)
    itime.name = 'time'
    result = pd.concat([iopen, ihigh, ilow, iclose, ivolume,
                        itime], axis=1).reset_index(drop=True)
    return result


def cusum(data, h):
    tEvents, sPos, sNeg = [], 0, 0
    diff = data.diff()
    for i in diff.index[1:]:
        sPos, sNeg = max(0, sPos+diff.loc[i]), min(0, sNeg+diff.loc[i])
        if sNeg < -h:
            sNeg = 0
            tEvents.append(i)
        elif sPos > h:
            sPos = 0
            tEvents.append(i)
    return pd.DatetimeIndex(tEvents)
