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
        self.profit_trades = 0
        self.loss_trades = 0
        # compute statistics
        cnt = len(self.trades)
        if cnt > 0:
            self.equity.apply(self.get_max_draw, axis=1)
            prof = self.trades[self.trades.pnl > 0]
            loss = self.trades[~(self.trades.pnl > 0)]
            std = self.trades.pnl.std()
            self.net_profit = self.trades.pnl.sum()
            self.gross_profit = prof.pnl.sum()
            self.gross_loss = loss.pnl.sum()
            self.max_draw = np.round(self.max_draw, 4)
            self.profit_factor = np.round(self.gross_profit / -self.gross_loss, 4) if self.gross_loss < 0 else None
            self.recovery_factor = np.round(self.net_profit / self.max_draw, 4) if self.max_draw > 0 else None
            self.sharpe_ratio = np.round(self.net_profit / std, 4) if std > 0 else None
            self.profit_trades = np.round(100.0 * len(prof) / cnt, 2)
            self.loss_trades = np.round(100.0 * len(loss) / cnt, 2)


    def get_max_draw(self, row):
        if row.pnl > self.max_pnl:
            self.max_pnl = row.pnl
            self.index_mpnl = row.name
        elif (self.max_pnl - row.pnl) > self.max_draw:
            self.max_draw = self.max_pnl - row.pnl
            self.index_mdraw = row.name


    def summary(self):
        if len(self.trades) > 0:
            print(f'Net profit:      {self.net_profit}')
            print(f'Gross profit:    {self.gross_profit}')
            print(f'Gross loss:      {self.gross_loss}')
            print(f'Max drawdown:    {self.max_draw}')
            print(f'Profit factor:   {self.profit_factor}')
            print(f'Recovery factor: {self.recovery_factor}')
            print(f'Sharpe ratio:    {self.sharpe_ratio}')
            print(f'Trades count:    {len(self.trades)}')
            print(f'Profitale:       {self.profit_trades}%')
            print(f'Losing:          {self.loss_trades}%')
        else:
            print('There is no trades here')


class Strategy:
    def reset(self):
        self.trades = []
        self.equity = []
        self.pnl_real = 0
        self.pnl_open = 0
        self.pos = 0
        self.price_enter = None
        self.index_enter = None

    def run(self, data, fee_lot=0, fee_pct=0):
        # setup fields
        self.reset()
        self.fee_lot = fee_lot
        self.fee_pct = fee_pct / 100.0
        # run strategy for each row
        data.apply(self.handle_next, axis=1)
        # close position at the end if present
        self.close(data.iloc[-1])
        # compute statistics
        return StrategyResult(trades=pd.DataFrame(self.trades),
                              equity=pd.DataFrame(self.equity, index=data.index))


    def handle_next(self, row):
        self.next(row)
        self.pnl_open = (self.pos * (row.close - self.price_enter)) if self.pos != 0 else 0
        self.equity.append({
            'pos': self.pos,
            'pnl': self.pnl_open + self.pnl_real,
            'trd_cnt': len(self.trades)
        })


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
            trd = {
                'pos': self.pos,
                'price_enter': self.price_enter,
                'price_exit': trd_price,
                'index_enter': self.index_enter,
                'index_exit': row.name,
                'fee': fee,
                'pnl': (self.pos * (trd_price - self.price_enter)) - fee
            }
            self.trades.append(trd)
            self.pnl_real += trd['pnl']
            self.pos = 0
            self.price_enter = None
            self.index_enter = None


    def enter(self, row, pos, trd_price):
        self.close(row, trd_price)
        self.pos = pos
        self.price_enter = trd_price
        self.index_enter = row.name


    def get_fee(self, price):
        return self.fee_lot + (self.fee_pct * price)



class InferenceStrategy(Strategy):
    def __init__(self, y_pred, min_change=0, min_change_pct=0):
        self.y_pred = y_pred
        self.min_change = min_change
        self.min_change_pct = min_change_pct / 100.0
        test1 = int(min_change > 0)
        test2 = int(min_change_pct > 0)
        if (test1 ^ test2) == 0:
            raise Exception('pass either `min_change` or `min_change_pct`')


    def next(self, row):
        ch = self.y_pred[row.name] - row.open
        min_ch = self.min_change + (row.open * self.min_change_pct)
        if (ch > min_ch):
            print(f'buy at {row.name} -> {row.open}')
            self.buy(row, row.open)
        elif (ch < -min_ch):
            print(f'sell at {row.name} -> {row.open}')
            self.sell(row, row.open)
        else:
            self.close(row, row.open)



class MeanReverseStrategy(Strategy):
    def __init__(self, y_pred, min_change=0, min_change_pct=0):
        self.y_pred = y_pred
        self.min_change = min_change
        self.min_change_pct = min_change_pct / 100.0
        test1 = int(min_change > 0)
        test2 = int(min_change_pct > 0)
        if (test1 ^ test2) == 0:
            raise Exception('pass either `min_change` or `min_change_pct`')


    def next(self, row):
        y_mid = self.y_pred[row.name]
        y_sell = round_pred(y_mid * (1.0 + self.min_change_pct) + self.min_change)
        y_buy = round_pred(y_mid * (1.0 - self.min_change_pct) - self.min_change)
        if (row.low <= y_buy):
            self.buy(row, y_buy)
            
        if (row.high >= y_sell):
            self.sell(row, y_sell)
        
        self.close(row)
