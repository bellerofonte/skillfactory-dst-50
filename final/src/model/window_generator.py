from tensorflow.keras.utils import Sequence

class WindowGenerator(Sequence):
    def __init__(self, X, y, window_size, batch_size, start=None, end=None):
        # get shapes and ranges
        self.n_samples, self.n_features = X.shape        
        self.window_size = window_size
        self.batch_size = batch_size if batch_size > 0 else self.n_samples
        self.start_idx = max(window_size, (start if start != None else 0))
        self.end_idx = min(self.n_samples, (end if end != None else self.n_samples))
        # get datasets
        self.X = X
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            self.get_x = self.__get_x_pandas
            self.index_x = X.index[self.start_idx:self.end_idx]
        elif isinstance(X, np.ndarray):
            self.get_x = self.__get_x_numpy
            self.index_x = None
        else:
            raise Exception('Not supported X type')
            
        self.y = y
        if isinstance(y, pd.Series):
            self.get_y = self.__get_y_pandas
            self.index_y = y.index[self.start_idx:self.end_idx]
        elif isinstance(y, np.ndarray):
            self.get_y = self.__get_y_numpy
            self.index_y = None
        else:
            self.get_y = self.__get_y_zero
            self.index_y = None
            
    
    def get_index(self):
        return self.index_y if self.index_x is None else self.index_x
    
    
    def __len__(self):
        return int(np.ceil((self.end_idx - self.start_idx) / self.batch_size))
    
    
    
    def __getitem__(self, idx):
        start = (idx * self.batch_size) + self.start_idx
        end = min(start + self.batch_size, self.end_idx) 
        batch_size = end - start
        X_batch = np.reshape(self.get_x(start, end), (batch_size, self.window_size, self.n_features))
        y_batch = self.get_y(start, end)
        return X_batch, y_batch
    
    
    def __get_x_pandas(self, start, end):
        return [self.X.iloc[i - self.window_size:i].to_numpy('float32') for i in range(start, end)]
    
    
    def __get_x_numpy(self, start, end):
        return [self.X[i - self.window_size:i,:] for i in range(start, end)]
    
    
    def __get_y_pandas(self, start, end):
        return self.y.iloc[start:end].to_numpy('float32')
    
    
    def __get_y_numpy(self, start, end):
        return self.y[start:end]
    
    
    def __get_y_zero(self, start, end):
        return np.zeros(end-start)
    
    