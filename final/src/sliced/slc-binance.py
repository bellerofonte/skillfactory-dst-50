import pandas as pd
from slicer import slice_ticks
import zipfile
import os

TICKS_PATH = os.path.abspath(os.path.dirname(__file__) + '../../../ticks/binance')
BARS_PATH = os.path.abspath(os.path.dirname(__file__) + '../../../bars')
SYMBOLS = [{
    'symbol': 'BTCUSDT',
    'price_step': 0.5,
    'size_step': 1
}, {
    'symbol': 'ETHUSDT',
    'price_step': 0.05,
    'size_step': 1
}]
FREQ = ['10s', '1min', '5min', '10min']

def read_file(sym, zip_name, freq):
    symbol = sym['symbol']
    filename = f'{TICKS_PATH}/{symbol}/{zip_name}'
    print(f'reading \'{filename}\'... ', end='')
    extention = filename[-4:]
    if extention == '.zip':
        csv_name = zip_name.replace('.zip', '.csv')
        zf = zipfile.ZipFile(filename) 
        df = pd.read_csv(zf.open(csv_name), header=None)
    elif extention == '.csv':        
        df = pd.read_csv(filename)
    else:
        print('Unsupported extention')
        return None

    no_idx = len(df.columns) == 6
    if no_idx:
        df.columns = ['id', 'price', 'size', 'quote_size', 'timestamp', 'is_sell']
    else:
        df.columns = ['idx', 'id', 'price', 'size', 'quote_size', 'timestamp', 'is_sell']

    df.drop_duplicates(subset='id', ignore_index=True, inplace=True)
    df.timestamp = pd.to_datetime(df.timestamp, unit='ms')
    df['side'] = 1 - (2 * df.is_sell.astype('int'))
    if no_idx:
        df.drop(columns=['id', 'quote_size', 'is_sell'], inplace=True)
    else:
        df.drop(columns=['idx', 'id', 'quote_size', 'is_sell'], inplace=True)
        
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
        filename = f'{BARS_PATH}/binance-{symbol}-{f}.csv'
        print(f'writing output to \'{filename}\'')
        df_all.to_csv(filename, date_format='%Y-%m-%d %H:%M:%S', index=False, compression='zip')

print('done')