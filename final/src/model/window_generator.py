import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence

class WindowGenerator(Sequence):
    def __init__(self, X, y, window_size, batch_size, start=None, end=None):
        lx = len(X)
        self.X = X
        self.y = y if isinstance(y, pd.Series) else pd.Series(data=np.zeros(lx), index=X.index)
        self.window_size = window_size
        self.batch_size = batch_size if batch_size > 0 else lx
        self.start_idx = max(window_size, (start if start != None else 0))
        self.end_idx = min(lx, (end if end != None else lx))
        self.n_features = X.shape[1]
        
    
    def __len__(self):
        return int(np.ceil((self.end_idx - self.start_idx) / self.batch_size))
    
    
    def __getitem__(self, idx):
        start = (idx * self.batch_size) + self.start_idx
        end = min(start + self.batch_size, self.end_idx) 
        batch_size = end - start
        X_batch = [self.X.iloc[i - self.window_size:i].to_numpy('float32') for i in range(start, end)]
        X_batch = np.reshape(X_batch, (batch_size, self.window_size, self.n_features))
        y_batch = self.y.iloc[start:end].to_numpy('float32')
        return X_batch, y_batch