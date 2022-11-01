import pandas as pd
from slicer import slice_ticks
import os

TICKS_PATH = os.path.abspath(os.path.dirname(__file__) + '../../../ticks/bitmex')
BARS_PATH = os.path.abspath(os.path.dirname(__file__) + '../../../bars')
SYMBOLS = [{
    'symbol': 'XBTUSD',
    'price_step': 0.5,
    'size_step': 1
}, {
    'symbol': 'ETHUSD',
    'price_step': 0.05,
    'size_step': 1
}]
FREQ = ['10s', '1min', '5min', '10min']

def read_file(sym, csv_name, freq):
    symbol = sym['symbol']
    filename = f'{TICKS_PATH}/{symbol}/{csv_name}'
    print(f'reading \'{filename}\'... ', end='')
    df = pd.read_csv(filename)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
    if isinstance(freq, str):
        res = slice_ticks(df, freq, price_step=sym['price_step'], size_step=sym['size_step'])
        print(res.shape)
        return res
    elif isinstance(freq, list):
        res = {}
        for f in freq:
            res[f] = slice_ticks(df, f, price_step=sym['price_step'], size_step=sym['size_step'])

        print([(f, *res[f].shape) for f in freq])
        return res
    else:
        print('Unsupported freq type')
        return None


for sym in SYMBOLS:
    symbol = sym['symbol']
    files = os.listdir(f'{TICKS_PATH}/{symbol}')
    files.sort()
    dfs = [read_file(sym, f, FREQ) for f in files]
    for f in FREQ:
        dfs_f = [d1[f] for d1 in dfs]
        df_all = pd.concat(dfs_f, ignore_index=True, axis=0) if len(dfs_f) > 0 else pd.DataFrame()
        df_all.sort_values(by='time', ignore_index=True, inplace=True)
        filename = f'{BARS_PATH}/bitmex-{symbol}-{f}.csv.zip'
        print(f'writing output to \'{filename}\'')
        df_all.to_csv(filename, date_format='%Y-%m-%d %H:%M:%S', index=False, compression='zip')

print('done!')