import pandas as pd
import numpy as np

def slice_ticks(data, freq):
    temp = data.timestamp.dt.floor(freq=freq)
    items = []
    for key, rows in data.groupby(temp):
        net_size = rows['size'] * rows['side']
        open = rows.price.iloc[0]
        close = rows.price.iloc[-1]
        upper = max(open, close)
        lower = min(open, close)
        up_idx = rows[rows.price > upper].index
        md_idx = rows[(rows.price <= upper) & (rows.price >= lower)].index
        lo_idx = rows[rows.price < lower].index
        item = {
            'time': key,
            'open': open,
            'high': rows.price.max(),
            'low': rows.price.min(),
            'close': close,
            'sum_vol': np.round(net_size.abs().sum(), 3),
            'net_vol': np.round(net_size.sum(), 3),
            'up_sum_vol': np.round(net_size[up_idx].abs().sum(), 3),
            'up_net_vol': np.round(net_size[up_idx].sum(), 3),
            'md_sum_vol': np.round(net_size[md_idx].abs().sum(), 3),
            'md_net_vol': np.round(net_size[md_idx].sum(), 3),
            'lo_sum_vol': np.round(net_size[lo_idx].abs().sum(), 3),
            'lo_net_vol': np.round(net_size[lo_idx].sum(), 3),
            'cnt': net_size.count()
        }
        if abs(item['sum_vol'] - item['up_sum_vol'] - item['md_sum_vol'] - item['lo_sum_vol']) > 0.000001:
            print('something wrong:', key)
        
        items.append(item)

    return pd.DataFrame(items)