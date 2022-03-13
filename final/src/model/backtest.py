import pandas as pd
import numpy as np

def round_pred(y_pred, minstep = 0.5):
    return np.round(y_pred / minstep) * minstep

class StrategyResult:
    
    def __init__(self, trades, equity):
        # init variables
        self.trades = trades
        self.equity = equity
        self.max_pnl = 0
        self.max_draw = 0
        self.index_mpnl = None
        self.index_mdraw = None
        self.net_profit = 0
        self.gross_profit = 0
        self.gross_loss = 0
        self.profit_factor = 0
        self.recovery_factor = 0
        self.sharpe_ratio = 0
        self.my_ratio = 0
        self.profit_trades = 0
        self.loss_trades = 0
        self.raw_profit = 0
        self.ppt = 0
        self.pppt = 0
        self.fee = 0
        # compute statistics
        cnt = len(trades)
        if cnt > 0:
            temp = trades.result
            avg_price = np.mean(trades.price_enter + trades.price_exit) / 2.0
            self.equity.apply(self.get_max_draw, axis=1)
            prof = temp[temp > 0]
            loss = temp[~(temp > 0)]
            std = temp.std()
            self.net_profit = temp.sum()
            self.raw_profit = self.trades.pnl.sum()
            self.fee = self.trades.fee.sum()
            self.gross_profit = prof.sum()
            self.gross_loss = loss.sum()
            self.max_draw = self.max_draw
            self.profit_factor = (self.gross_profit / -self.gross_loss) if self.gross_loss < 0 else np.inf
            self.recovery_factor = (self.net_profit / self.max_draw) if self.max_draw > 0 else np.inf
            self.sharpe_ratio = (self.net_profit / std) if std > 0 else np.inf
            self.profit_trades = (100.0 * len(prof) / cnt)
            self.loss_trades = (100.0 * len(loss) / cnt)
            self.ppt = self.net_profit / cnt
            self.pppt = 100.0 * self.ppt / avg_price
            self.my_ratio = np.sign(self.net_profit) * np.log(1 + (self.sharpe_ratio * self.pppt))


    def get_max_draw(self, row):
        if row.pnl > self.max_pnl:
            self.max_pnl = row.pnl
            self.index_mpnl = row.name
        elif (self.max_pnl - row.pnl) > self.max_draw:
            self.max_draw = self.max_pnl - row.pnl
            self.index_mdraw = row.name


    def summary(self):
        trd_cnt = len(self.trades)
        if trd_cnt > 0:
            print(f'Net profit:      {self.net_profit:12.2f}     Gross profit:    {self.gross_profit:12.2f}')
            print(f'Raw profit:      {self.raw_profit:12.2f}     Gross loss:      {self.gross_loss:12.2f}')
            print(f'Fee paid:        {self.fee:12.2f}     Max drawdown:    {self.max_draw:12.2f}')
            print(f'Trades count:    {trd_cnt:12}     Profit factor:   {self.profit_factor:12.2f}')
            print(f'Profitable:      {self.profit_trades:11.2f}%     Recovery factor: {self.recovery_factor:12.2f}')
            print(f'Losing:          {self.loss_trades:11.2f}%     Sharpe ratio:    {self.sharpe_ratio:12.2f}')
            print(f'PPT %:           {self.pppt:11.2f}%     My ratio:        {self.my_ratio:12.2f}')
        else:
            print('There is no trades here')
            
    
    def best_pos(self, min_bars_held=0):
        def fix_trade(t, target, mbh):
            enter = t.index_enter
            imax = t.index_max
            start = imax + 1
            end = t.index_exit
            if start < end:
                target.loc[start:end] = 0
                
            if (imax - enter) < mbh:
                target.loc[enter:imax] = 0

        fix_pos = self.equity.pos.copy()
        self.trades.apply(fix_trade, axis=1, target=fix_pos, mbh = min_bars_held)
        return fix_pos
    

class Strategy:    
    def reset_pos(self):
        self.pnl_open = 0
        self.pnl_open_max = 0
        self.pnl_open_min = 0
        self.pos = 0
        self.price_enter = None
        self.index_enter = None
        self.index_open_max = None
        self.index_open_min = None
        
    
    def reset(self):
        self.trades = []
        self.equity = []
        self.pnl_real = 0
        self.reset_pos()

        
    def run(self, data, fee_lot=0, fee_pct=0):
        # setup fields
        self.reset()
        self.fee_lot = fee_lot
        self.fee_pct = fee_pct / 100.0
        # pre-compute everything you need
        self.prepare(data)
        # run strategy for each data row
        data.apply(self.handle_next, axis=1)
        # close position at the end if present
        if self.pos != 0:
            self.close(data.iloc[-1])
            self.equity[-1] = {
                'pos': 0,
                'pnl': self.pnl_real,
                'pnl_open': 0,
                'pnl_open_max': 0,
                'pnl_open_min': 0,
                'trd_cnt': len(self.trades)
            }
        # compute statistics
        return StrategyResult(pd.DataFrame(self.trades),
                              pd.DataFrame(self.equity, index=data.index))


    def handle_next(self, row):
        self.next(row)
        self.pnl_open = (self.pos * (row.close - self.price_enter)) if self.pos != 0 else 0
        if (self.pnl_open > self.pnl_open_max):
            self.pnl_open_max = self.pnl_open
            self.index_open_max = row.name
            
        if (self.pnl_open < self.pnl_open_min):
            self.pnl_open_min = self.pnl_open
            self.index_open_min = row.name
        
        self.equity.append({
            'pos': self.pos,
            'pnl': self.pnl_open + self.pnl_real,
            'pnl_open': self.pnl_open,
            'pnl_open_max': self.pnl_open_max,
            'pnl_open_min': self.pnl_open_min,
            'trd_cnt': len(self.trades)
        })
        

    def prepare(self, data):
        raise NotImplementedError('This method is not implemented')

        
    def next(self, row):
        raise NotImplementedError('This method is not implemented')


    def buy(self, row, buy_price):
        if (row.low <= buy_price and self.pos != 1):
            trd_price = min(buy_price, row.high)            
            self.enter(row, 1, trd_price)
            

    def sell(self, row, sell_price):
        if (row.high >= sell_price and self.pos != -1):
            trd_price = max(sell_price, row.low)
            self.enter(row, -1, trd_price)
            
            
    def close(self, row, close_price = None):
        if self.pos != 0:
            trd_price = row.close if close_price == None else close_price
            fee = self.get_fee(self.price_enter) + self.get_fee(trd_price)
            pnl = self.pos * (trd_price - self.price_enter)
            trd = {
                'pos': self.pos,
                'price_enter': self.price_enter,
                'price_exit': trd_price,
                'index_enter': self.index_enter,
                'index_exit': row.name,
                'index_max': self.index_open_max,
                'index_min': self.index_open_min,
                'fee': fee,
                'pnl': pnl,
                'pnl_max': self.pnl_open_max,
                'pnl_min': self.pnl_open_min,
                'result': pnl - fee 
            }
            self.trades.append(trd)
            self.pnl_real += trd['pnl']
            self.reset_pos()


    def enter(self, row, pos, trd_price):
        self.close(row, trd_price)
        self.pos = pos
        self.price_enter = trd_price
        self.index_enter = row.name
        self.index_open_max = row.name
        self.index_open_min = row.name
        

    def get_fee(self, price):
        return self.fee_lot + (self.fee_pct * price)

            
class SignalStrategy(Strategy):
    def __init__(self, signal, shift=0, price_name='open'):
        self.signal = signal
        self.shift = shift
        self.price_name = price_name
        
    def prepare(self, data):
        if type(self.signal) == str:
            if not self.signal in data.columns:
                raise Exception('missing signal column')
                
            self.shift = 0                
            self.next_signal = self.next_str
        elif isinstance(self.signal, pd.Series):
            self.next_signal = self.next_series
        elif isinstance(self.signal, np.ndarray):
            self.signal = pd.Series(data=self.signal, index=data.index)
            self.next_signal = self.next_series
        elif callable(getattr(self.signal, 'predict', None)):
            self.signal = pd.Series(data=self.signal.predict(data), index=data.index)
            self.next_signal = self.next_series
        else:
            raise Exception('unsupported signal type')
        
        if self.shift != 0:
            self.signal = self.signal.shift(self.shift).fillna(0)
            
        if not self.price_name in data.columns:
            raise Exception('missing price column')
    
    
    def next_str(self, row):
        return row[self.signal]
    
    
    def next_series(self, row):
        return self.signal[row.name]

    
    def next(self, row):
        signal = self.next_signal(row)
        price = row[self.price_name]
        if signal > 0:
            self.buy(row, price)
        elif signal < 0:
            self.sell(row, price)
        else:
            self.close(row, price)
