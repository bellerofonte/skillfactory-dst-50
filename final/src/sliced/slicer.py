import pandas as pd
import numpy as np

PRICE_DEC = 1
SIZE_DEC = 3

def slice_ticks(data, freq):
    temp = data.timestamp.dt.floor(freq=freq)
    items = []
    for key, rows in data.groupby(temp):
        # grab some columns
        r_size = rows['size']
        r_price = rows['price']
        r_side = rows['side']
        # compute some basic stats
        sum_vol = r_size.sum()
        net_size = r_size * r_side
        open = r_price.iloc[0]
        close = r_price.iloc[-1]
        upper = max(open, close)
        lower = min(open, close)
        # split items into groups
        up_idx = rows[r_price > upper].index
        md_idx = rows[(r_price <= upper) & (r_price >= lower)].index
        lo_idx = rows[r_price < lower].index
        # buy_rows = rows[r_side > 0]
        # sell_rows = rows[r_side < 0]
        item = {
            'time': key,
            'open': open,
            'high': r_price.max(),
            'low': r_price.min(),
            'close': close,
            'avg': np.round(r_price.mean(), PRICE_DEC),
            'wavg': np.round((r_price * r_size).sum() / sum_vol, PRICE_DEC),
            'median': r_price.median(),
            # 'buy_avg': np.round(buy_rows['price'].mean(), PRICE_DEC),
            # 'buy_wavg': np.round((buy_rows['price'] * buy_rows['size']).sum() / buy_rows['size'].sum(), PRICE_DEC),
            # 'buy_median': buy_rows.price.median(),
            # 'sell_avg': np.round(sell_rows['price'].mean(), PRICE_DEC),
            # 'sell_wavg': np.round((sell_rows['price'] * sell_rows['size']).sum() / sell_rows['size'].sum(), PRICE_DEC),
            # 'sell_median': sell_rows.price.median(),
            'sum_vol': np.round(sum_vol, 3),
            'up_sum_vol': np.round(r_size[up_idx].sum(), 3),
            'md_sum_vol': np.round(r_size[md_idx].sum(), 3),
            'lo_sum_vol': np.round(r_size[lo_idx].sum(), 3),
            'net_vol': np.round(net_size.sum(), 3),
            'up_net_vol': np.round(net_size[up_idx].sum(), 3),
            'md_net_vol': np.round(net_size[md_idx].sum(), 3),
            'lo_net_vol': np.round(net_size[lo_idx].sum(), 3),
            'cnt': r_price.count(),
            'hbl': (r_price.idxmax() < r_price.idxmin())
        }
        if abs(item['sum_vol'] - item['up_sum_vol'] - item['md_sum_vol'] - item['lo_sum_vol']) > 0.000001:
            print('something wrong:', key)
        
        items.append(item)

    return pd.DataFrame(items)