# %%
import pandas as pd
import requests as rq
import time
import os
import hashlib
import hmac
from apikey_bitmex import API_KEY, API_SECRET

BASE_URL = 'https://www.bitmex.com'
DATE_FROM = pd.Timestamp('2022-01-30')
DATE_TILL = pd.Timestamp('2022-08-04')
SYMBOLS = ['XBTUSD', 'ETHUSD']
PATH = os.path.abspath(os.path.dirname(__file__) + '../../../ticks/bitmex')
DEFAULT_WAIT = 0.5

# %%
def get_chunk(symbol, date, start):
    st = date.strftime('%Y-%m-%d')
    et = (date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    try:
        query = f'/api/v1/trade?symbol={symbol}&count=1000&start={start}&reverse=false&startTime={st}&endTime={et}'
        expires = str(int(round(time.time()) + 5))
        message = bytes('GET' + query + expires, 'utf-8')
        signature = hmac.new(bytes(API_SECRET, 'utf-8'), message, digestmod=hashlib.sha256).hexdigest()
        headers= {
            'api-expires': expires,
            'api-key': API_KEY,
            'api-signature': signature
        }
        res = rq.get(BASE_URL + query, headers=headers)
        json = res.json()
        if res.status_code == 200:
            print(f'\r{symbol}:{st}:{start} -> {len(json)} rows', end='')
            return json
        else:
            msg = json['error']['message']
            print(f'\r{symbol}:{st}:{start} -> {msg}', end='')
            return None
    except Exception as e:
        print(f'\r{symbol}:{st}:{start} -> error: {e}', end='')
        return None


# %%
def get_day(symbol, date):
    wait_time = DEFAULT_WAIT
    err_cnt = 0
    res = []
    start = 0
    ds = date.strftime('%Y-%m-%d')
    while True:
        chunk = get_chunk(symbol, date, start)
        if chunk == None:
            err_cnt += 1
            wait_time = min(wait_time + 20, wait_time * 2)
        else:
            err_cnt = 0
            wait_time = DEFAULT_WAIT
            res.extend(chunk)
            start += len(chunk)
            if len(chunk) == 0: # no more data for that day
                print(f'\r{symbol}:{ds} -> {start} rows.    ')
                return res

        if err_cnt >= 10:
            print(f'\r{symbol}:{ds} -> too many errors.    ')
            return None
    
        time.sleep(wait_time)


# %%
def make_df(items):
    df = pd.DataFrame(items)
    df.drop_duplicates(subset='trdMatchID', inplace=True)
    df.timestamp = pd.to_datetime(df.timestamp)
    df.side = df.side.apply(lambda s: 1 if s == 'Buy' else -1)
    df = df[['timestamp', 'price', 'size', 'side']]
    return df

# %%
def save_day(symbol, date, items):
    ds = date.strftime('%Y-%m-%d')
    filename = f'{PATH}/{symbol}/{ds}.csv'
    df = make_df(items)
    df.to_csv(filename, date_format='%Y-%m-%d %H:%M:%S.%f', index=False)

# %%
for symbol in SYMBOLS:
    os.makedirs(f'{PATH}/{symbol}', exist_ok=True)
    
date = DATE_FROM
while date <= DATE_TILL:
    for symbol in SYMBOLS:
        items = get_day(symbol, date)
        save_day(symbol, date, items)

    date = date + pd.Timedelta(days=1)

# %%



