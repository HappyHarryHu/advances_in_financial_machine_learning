import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count, RLock
from pandas.core.algorithms import unique
from tqdm import tqdm
import gc


def _uniqueness_matrix(trange, start, end, line):
    out = pd.DataFrame(False, index=trange, columns=start)
    for s, e in tqdm(zip(start, end), total=len(start), position=line, leave=False):
        out.loc[s:e, s] = True
    out.columns = out.columns.strftime('%Y-%m-%d %H:%M:%S')
    out.to_parquet(f'temp/{line}.parquet')
    print(f'Part{line} finished')


def uniqueness_matrix(events, freq, core=cpu_count()):
    with Pool(core, initargs=(RLock(),), initializer=tqdm.set_lock) as pool:
        l = events.shape[0]//core
        args = []
        for i in range(core):
            events_ = events.iloc[i*l:(i+1)*l]
            start, end = events_.index.shift(freq=freq), events_[
                't1'].shift(freq=freq)
            trange = pd.date_range(start=start[0], end=end.max(), freq=freq)
            args.append((trange, start, end, i))
        pool.starmap(_uniqueness_matrix, args)            
    ind_matrices = []
    for i in range(core):
        print(f'Processing part{i}', end='\r')
        ind_matrices.append(pd.read_parquet(f'temp/{i}.parquet'))
    concurrent = pd.concat(map(lambda x: x.sum(1), ind_matrices), join='outer', axis=1).fillna(0).sum(1)
    del ind_matrices
    gc.collect()
    uniqueness = []
    for i in range(core):
        print(f'Processing part{i}', end='\r')
        ind_matrix = pd.read_parquet(f'temp/{i}.parquet')
        uniqueness.append((ind_matrix.mul(concurrent[ind_matrix.index], axis=0)).apply(lambda x: (1 / x[x > 0]).mean()).fillna(0))
    return pd.concat(uniqueness)


def sequential_bootstrap(ind_matrix, size=None):
    size = ind_matrix.shape[1] if size is None else size
    result = []
    for i in tqdm(range(size)):
        avg_uni = []
        for event in ind_matrix.columns:
            concurrency = ind_matrix[result +
                                     [event.strftime('%Y-%m-%d %H:%M:%S')]].sum(1)
            uni = ind_matrix[event].mul(concurrency, axis=0)
            avg_uni.append(uni[uni > 0].mean())
        t = np.random.choice(ind_matrix.columns,
                             p=np.array(avg_uni)/sum(avg_uni))
        result.append(pd.to_datetime(str(t)).strftime('%Y-%m-%d %H:%M:%S'))
    return pd.Series(result)


def sample_weights_return(close, ind_matrix):
    concurrent = ind_matrix.sum(1)
    log_ret = np.log(ind_matrix.mul(close, axis=0)) - \
        np.log(ind_matrix.mul(close, axis=0).shift(1))
    weights = np.abs(
        (log_ret.div(concurrent, axis=0)).replace([np.inf, -np.inf, np.nan], 0).sum())
    return weights / np.sum(weights)


def time_decay(weights, c=1):
    # apply piecewise-linear decay to observed uniqueness (tW)
    # newest observation gets weight=1, oldest observation gets weight=clfLastW
    weights = weights.sort_index().cumsum()
    if c >= 0:
        slope = (1 - c) / weights.iloc[-1] 
    else: 
        slope = 1 / ((c + 1) * weights.iloc[-1]) 
    const = 1 - slope * weights.iloc[-1] 
    weights = const + slope * weights 
    weights[weights < 0] = 0 
    return weights


if __name__ == '__main__':
    events = pd.read_csv('data/short_labeling.csv', index_col=0, parse_dates=True)
    events['t1'] = pd.to_datetime(events['t1'])
    uniqueness = uniqueness_matrix(events, '5min')
    uniqueness.to_csv('data/short_uniqueness.csv')
    # seq_boost_sample = sequential_bootstrap(ind_matrix)
    # seq_boost_sample.to_csv('seq_boost_sample.csv')
    # data = pd.read_csv('data/BTCSPOT_300.csv', index_col=0, parse_dates=True)
    # weights = sample_weights_return(data['close'], ind_matrix)
