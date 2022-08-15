import numpy as np
import pandas as pd

class NoneScaler:
    def __init__(self, x_type=np.float32):
        self.x_type = x_type
        
    def fit_transform(self, x):
        return x.to_numpy(self.x_type)
    
    
class MinMaxGroup:
    def __init__(self, columns, x_type=np.float32):
        self.columns = columns
        self.x_type = x_type
        self.index = None
        
        
    def set_index(self, index):
        self.index = index
        
        
    def fill(self, x, values):
        max_ = x[self.columns].max().max()
        min_ = x[self.columns].min().min()
        rng_ = max_ - min_
        for i in self.index:
            values[:,i - 1] = (x.iloc[:,i] - min_) / rng_ # '-1' because 0th column in pandas is 'index'
            
            
class MinMaxStdGroup:
    def __init__(self, columns, stdev_column, min_stdev=1.0, x_type=np.float32):
        self.columns = columns
        self.std_col = stdev_column
        self.min_std = min_stdev
        self.x_type = x_type
        self.index = None
        
        
    def set_index(self, index):
        self.index = index
        
        
    def fill(self, x, values):
        std_ = np.max(self.min_stdev, x[self.std_col].std())
        max_ = x[self.columns].max().max() / std_
        min_ = x[self.columns].min().min() / std_
        rng_ = max_ - min_
        for i in self.index:
            values[:,i - 1] = ((x.iloc[:,i] / std_) - min_) / rng_ # '-1' because 0th column in pandas is 'index'
            
            
class MinMaxLogGroup:
    def __init__(self, columns, x_type=np.float32):
        self.columns = columns
        self.x_type = x_type
        self.index = None
        
        
    def set_index(self, index):
        self.index = index
        
        
    def fill(self, x, values):
        max_ = np.log(1.0 + x[self.columns].max().max())
        min_ = np.log(1.0 + x[self.columns].min().min())
        rng_ = max_ - min_
        for i in self.index:
            values[:,i - 1] = (np.log(1.0 + x.iloc[:,i]) - min_) / rng_ # '-1' because 0th column in pandas is 'index'