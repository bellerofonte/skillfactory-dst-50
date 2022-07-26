from tensorflow.keras.utils import Sequence
import numpy as np
import pandas as pd

class WindowGenerator(Sequence):
    def __init__(self, X, y=None, start=None, end=None, window_size=10, batch_size=32, filter=None, 
                 shuffle=False, random_seed=None, x_type=np.float32, y_type=np.float32):
        # get shapes and ranges
        self.n_samples, self.n_features = X.shape        
        self.window_size = window_size
        self.batch_size = min(batch_size if batch_size > 0 else self.n_samples, self.n_samples)
        self.start_idx = max(window_size, (start if start != None else 0))
        self.end_idx = min(self.n_samples, (end if end != None else self.n_samples))
        self.x_type = x_type
        self.y_type = y_type
        self.index = None
        self.X = X
        self.y = y
        self.last_result = None
        
        # get datasets
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            if isinstance(y, np.ndarray):
                self.y = pd.Series(y, index=X.index)
            if not (y is None or isinstance(y, pd.Series)):
                raise Exception('The `y` should be either a `pandas.Series` or `numpy.ndarray` or `None`')
            
            if filter is None:
                self.index = X.index[self.start_idx:self.end_idx]
            elif isinstance(filter, pd.Series):
                fltr = filter.astype('bool').iloc[self.start_idx:self.end_idx]
                self.index = fltr[fltr == True].index
            else:
                raise Exception('`filter` should be either `None` or `pandas.Series`')
                
            self.__get_item_impl = self.__get_item_pandas
            self.iter = [i for i in range(0, len(self.index), self.batch_size)]
            
        elif isinstance(X, np.ndarray):
            if isinstance(y, pd.Series):
                self.y = y.to_numpy(y_type)
            elif not (y is None or isinstance(y, np.ndarray)):
                raise Exception('The `y` should be either a `pandas.Series` or `numpy.ndarray` or `None`')
                
            self.__get_item_impl = self.__get_item_numpy
            self.iter = [i for i in range(self.start_idx, self.end_idx, self.batch_size)]
        else:
            raise Exception('Unsupported `X` type')
            
        self.cnt = len(self.iter)
        if (shuffle):
            rng = np.random.default_rng(random_seed)
            rng.shuffle(self.iter)
            
    
    def get_index(self):
        return self.index
    
    
    def get_shape(self):
        return (self.window_size, self.n_features)
        
    
    def __len__(self):
        return self.cnt
    
    
    # def __getitem__(self, idx):
    #     start = self.iter[idx]
    #     end = min(start + self.batch_size, self.end_idx) 
    #     batch_size = end - start
    #     X_batch = np.reshape(self.get_x(start, end), (batch_size, self.window_size, self.n_features))
    #     y_batch = self.get_y(start, end)
    #     return X_batch, y_batch
    
    
    def __getitem__(self, idx):
        return self.__get_item_impl(idx)
        
    
    def __get_item_pandas(self, idx):
        start = self.iter[idx]
        end = min(start + self.batch_size, len(self.index))
        batch_size = end - start
        try:
            X_batch = [self.X.loc[(i - self.window_size + 1):i].to_numpy(self.x_type) for i in self.index[start:end]]
            X_batch = np.reshape(X_batch, (batch_size, self.window_size, self.n_features))
            y_batch = self.y.loc[self.index[start:end]].to_numpy(self.y_type) if not self.y is None else None
            return X_batch, y_batch
        except Exception as ex:
            raise Exception('Didn\'t you forget to reset the index?') from ex
            
    
    
    def __get_item_numpy(self, idx):
        start = self.iter[idx]
        end = min(start + self.batch_size, self.end_idx) 
        batch_size = end - start
        X_batch = [self.X[i - self.window_size:i,:] for i in range(start, end)]
        X_batch = np.reshape(X_batch, (batch_size, self.window_size, self.n_features))
        y_batch = self.y[start:end] if not self.y is None else None
        return X_batch, y_batch
    
    
#     def __get_x_pandas(self, start, end):
#         return [self.X.iloc[i - self.window_size:i].to_numpy(np.float32) for i in range(start, end)]
    
    
#     def __get_x_numpy(self, start, end):
#         return [self.X[i - self.window_size:i,:] for i in range(start, end)]
    
    
#     def __get_y_pandas(self, start, end):
#         return self.y.iloc[start:end].to_numpy(self.y_type)
    
    
#     def __get_y_numpy(self, start, end):
#         return 
    
    
#     def __get_y_zero(self, start, end):
#         return np.zeros(end-start)
    
    
class MergeWindowGenerator(Sequence):
    def __init__(self, items, shuffle=False, random_seed=None):
        if not isinstance(items, (list, tuple, np.ndarray)):
            raise Exception('`items` should be iterable')
    
        self.items = items
        self.iter = np.concatenate([[(i, j) for j in range(len(item))] for i, item in enumerate(items)])
        if (shuffle):
            rng = np.random.default_rng(random_seed)
            rng.shuffle(self.iter)
            
        
    def __len__(self):
        return len(self.iter)
    
    
    def __getitem__(self, idx):
        i, j = self.iter[idx]
        return (self.items[i])[j]
    
    
    
def create_gens(X, y, test_frac=0.1, val_frac=0.1, **kwargs):
    sz = X.shape[0]
    test_size = int(sz * test_frac)
    val_size = int(sz * val_frac)
    train_size = sz - val_size - test_size
    
    train_gen = WindowGenerator(X, y, end=train_size, **kwargs)
    val_gen = WindowGenerator(X, y, start=train_size, end=(train_size + val_size), **kwargs)
    test_gen = WindowGenerator(X, y, start=(train_size + val_size), **kwargs)
    
    return train_gen, val_gen, test_gen
