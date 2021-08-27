import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count, RLock
from tqdm import tqdm
from datetime import timedelta


def _triple_barrier(close, events, ptSl, molecule, line):
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    # events: [t1: vertical barrier, trgt: width of horizontal barrier, side: position direction]
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    if ptSl[0] > 0:
        pt = ptSl[0]*events['trgt']
    else:
        pt = pd.Series(index=events_.index)  # NaNs
    if ptSl[1] > 0:
        sl = -ptSl[1]*events['trgt']
    else:
        sl = pd.Series(index=events_.index)  # NaNs
    for loc, t1 in tqdm(events_['t1'].fillna(close.index[-1]).iteritems(), desc=f'Since {molecule[0]}', position=line, total=len(molecule), leave=False):
        df0 = close[loc:t1].copy()  # path prices
        df0 = (df0/close[loc]-1)*events_.at[loc, 'side']  # path returns
        # earliest stop loss.
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()
        # earliest profit taking.
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()
    return out


def triple_barrier(data, t1, side, ptSl, core=cpu_count(), min_ret=0):
    events = pd.DataFrame(index=data.index)
    events['t1'] = events.index.shift(freq=timedelta(hours=t1))
    log_ret = np.log(data['close'].shift(
        freq=timedelta(hours=-t1))) - np.log(data['close'])
    events['trgt'] = log_ret.rolling(f'{t1*100}H').std()
    events['side'] = side
    with Pool(core, initargs=(RLock(),), initializer=tqdm.set_lock) as pool:
        l = events.shape[0] // core
        molecules = [events.iloc[i*l:(i+1)*l].index for i in range(core)]
        args = list((data['close'], events, ptSl, m, i)
                    for i, m in enumerate(molecules))
        result = sorted(pool.starmap(_triple_barrier, args),
                        key=lambda x: x.index[0])
    t1 = pd.concat(result).min(axis=1)
    t1 = t1[t1.isin(data.index)]
    trgt = events.loc[t1.index, 'trgt']
    ret = pd.Series(np.log(data.loc[t1, 'close'].values / data.loc[t1.index,
                    'close'].values), index=t1.index) * events.loc[t1.index, 'side']
    bin = ret > min_ret
    return pd.concat({'t1': t1, 'trgt': trgt, 'ret': ret, 'bin': bin}, axis=1)


if __name__ == '__main__':
    ohlcv = pd.read_csv('BTCSPOT_300.csv', index_col=0, parse_dates=True)
    side_labeling = triple_barrier(ohlcv, 1, 1, (3, 3))
    side_labeling.to_csv('side_labeling.csv')
    long_labeling = triple_barrier(ohlcv, 1, 1, (3, 1), min_ret=0.00065)
    long_labeling.to_csv('long_labeling.csv')
    short_labeling = triple_barrier(ohlcv, 1, -1, (3, 1), min_ret=0.00065)
    short_labeling.to_csv('short_labeling.csv')