import numpy as np
import pandas as pd
from datetime import timedelta


def train_test_split(data, train, test, holding):
    train_window = timedelta(days=train)
    refit_window = timedelta(days=test)
    skip_window = timedelta(hours=holding)

    index_ls = []
    curr = data.index[0]
    while curr + train_window < data.index[-1]:
        index_ls.append((
            data[curr: curr+train_window-skip_window].index,
            data[curr+train_window: curr+train_window+refit_window].index
        ))
        curr += refit_window
    return index_ls


def gen_dataset(data):    
    # Momentum Signals [price, volume, combined]
    def _mmt_sigs(close, volume):
        def _momentum(x):
            ema8 = x.ewm(alpha=1/8).mean()
            ema24 = x.ewm(alpha=1/24).mean()
            ema72 = x.ewm(alpha=1/72).mean()
            ema168 = x.ewm(alpha=1/168).mean()
            ema504 = x.ewm(alpha=1/504).mean()

            def _mean_rev_func(x):
                return x * np.exp(-x.pow(2)/4) / (np.sqrt(2) * np.exp(-0.5))

            x1 = ema8 - ema24
            z1 = x1/x.rolling(168).std()
            z1 = z1/z1.rolling(720).std()
            z1 = _mean_rev_func(z1)

            x2 = ema24 - ema72
            z2 = x2/x.rolling(168).std()
            z2 = z2/z2.rolling(720).std()
            z2 = _mean_rev_func(z2)

            x3 = ema168 - ema504
            z3 = x3/x.rolling(168).std()
            z3 = z3/z3.rolling(720).std()
            z3 = _mean_rev_func(z3)

            return (z1 + z2 + z3) / 3

        signed_volume = volume * np.sign(close.diff())
        signed_volume_ma = signed_volume.rolling(8).sum().ffill().dropna()

        price_sig = _momentum(close)
        volume_sig = _momentum(signed_volume_ma)

        combined_sig = (
            price_sig * abs(price_sig) + volume_sig * abs(volume_sig)
        ) / (abs(price_sig) + abs(volume_sig))

        out = pd.DataFrame(
            {
                'price_sig': price_sig,
                'volume_sig': volume_sig,
                'combined_sig': combined_sig
            }
        )
        return out
        
    # Standard Features [m1, m2, ...]
    def _std(close, volume):
        m1 = close / close.ewm(span=3*12).mean() - 1
        m2 = close / close.ewm(span=6*12).mean() - 1
        vm1 = close / (
            (volume * close).rolling(3*12).sum()
            / volume.rolling(3*12).sum()
        ) - 1
        vm2 = close / (
            (volume * close).rolling(6*12).sum()
            / volume.rolling(6*12).sum()
        )
        _high = close.rolling(12*12).max()
        _low = close.rolling(12*12).min()
        k = (close - _low) / (_high - _low + 1e-5)
        d = k.rolling(3*12).std()
        _rtn = close.pct_change()
        max_12 = _rtn.rolling(12*12).max()
        min_12 = _rtn.rolling(12*12).min()
        sr = close.pct_change(6*12) / (_rtn.rolling(6*12).std() * 6*12)
        sr_long = close.pct_change(12*12) / (_rtn.rolling(12*12).std() * 12*12)
        
        out = pd.DataFrame(
            {
                'm1_5m': m1, 'm2_5m': m2,
                'vm1_5m': vm1, 'vm2_5m': vm2,
                'k_5m': k, 'd_5m': d,
                'min_12_5m': min_12, 'max_12_5m': max_12,
                'sr_5m': sr, 'sr_long_5m': sr_long,
            }
        )
        return out
    
    dataset = pd.concat(
        [
            _mmt_sigs(data['close'], data['volume']),
            _std(data['close'], data['volume']),
        ], 1
    )

    dataset.fillna(method='ffill', inplace=True)
    dataset.dropna(inplace=True)
    
    return dataset
