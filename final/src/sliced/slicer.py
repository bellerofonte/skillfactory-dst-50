import pandas as pd
import numpy as np

PRICE_DEC = 1
SIZE_DEC = 3

def slice_ticks(data, freq, price_step=0.5, size_step=1.0, price_dec=PRICE_DEC, size_dec=SIZE_DEC, hist=True):
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
        net_vol = net_size.sum()
        open = r_price.iloc[0]
        high = r_price.max()
        low = r_price.min()
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
            'high': high,
            'low': low,
            'close': close,
            'avg': np.round(r_price.mean(), price_dec),
            'wavg': np.round((r_price * r_size).sum() / sum_vol, price_dec),
            'median': r_price.median(),
            'sum_vol': np.round(sum_vol, 3),
            'up_sum_vol': np.round(r_size[up_idx].sum(), 3),
            'md_sum_vol': np.round(r_size[md_idx].sum(), 3),
            'lo_sum_vol': np.round(r_size[lo_idx].sum(), 3),
            'net_vol': np.round(net_vol, 3),
            'up_net_vol': np.round(net_size[up_idx].sum(), 3),
            'md_net_vol': np.round(net_size[md_idx].sum(), 3),
            'lo_net_vol': np.round(net_size[lo_idx].sum(), 3),
            'oto': np.round(net_vol / sum_vol, 2),
            'cnt': r_price.count(),
            'hbl': (r_price.idxmax() < r_price.idxmin())
        }
        if abs(item['sum_vol'] - item['up_sum_vol'] - item['md_sum_vol'] - item['lo_sum_vol']) > 0.000001:
            print('something wrong:', key)

        if hist:
            def make_vol_hist(row, vhb, vhs, price_step, vol_step):
                side_ = int(row['side'] > 0) # 1 for buy and 0 for sell
                size_ = int(row['size'] / vol_step)
                idx_ = int((row['price'] - low) / price_step)
                vhb[idx_] += (size_ * side_)
                vhs[idx_] += (size_ * (1 - side_))

            hist_len = 1 + int((high - low) / price_step)
            hist_b = np.zeros(hist_len, np.int32)
            hist_s = np.zeros(hist_len, np.int32)
            rows.apply(lambda r: make_vol_hist(r, hist_b, hist_s, price_step, size_step), axis=1)
            item['vhb'] = hist_b
            item['vhs'] = hist_s
        
        items.append(item)

    return pd.DataFrame(items)