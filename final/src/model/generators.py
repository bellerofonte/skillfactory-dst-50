from tensorflow.keras.utils import Sequence
import numpy as np
import pandas as pd

class WindowGenerator(Sequence):
    '''
    Creates (on the fly) windows with specified size from given dataset 
    '''
    
    def __init__(self, X, y=None, start=None, end=None, window_size=100, batch_size=32, columns=None,
                 filter=None, shuffle=False, random_seed=None, x_type=np.float32, y_type=np.float32):
        '''
        Constructor

            Parameters:
                X (pandas.DataFrame|pandas.Series|numpy.ndarray): features
                y (pandas.Series|numpy.ndarray|str|None): target
                    If `pandas.Series` or `numpy.ndarray` - supposed t have same rows count as X and only 1 column
                    If `str` - considered a column name inside X
                    If `None` - not used (inference)
                start (int|None): if not `None` considered a number of the first used row (range style, not index!)
                end (int|None): if not `None` considered a number of next to the last used row (range style, not index!)
                window_size (int): window size for batch generation
                batch_size (int): number of windows in one batch
                columns: (list(str)|None): if not `None` considered a subset of columns used for batches 
                    applicable only if `X` is `pandas.DataFrame`
                filter: (pandas.Series|None): if not `None` - used only rows what have `True` value in within the `filter` 
                    applicable only if `X` is `pandas.DataFrame`
                shuffle (bool): shuffle or not 
                random_seed (int|None): random state for random numbers generator
                    applicable only if `shuffle` is `True`
                x_type (type): numpy data type for features
                y_type (type): numpy data type for target                

            Returns:
                    instance of class
        '''
        # get shapes and ranges
        self.X = X if columns is None else X[columns]
        self.y = X[y] if isinstance(y, str) else y
        self.n_samples, self.n_features = self.X.shape        
        self.window_size = window_size
        self.batch_size = min(batch_size if batch_size > 0 else self.n_samples, self.n_samples)
        self.start_idx = max(window_size, (0 if start is None else start))
        self.end_idx = min(self.n_samples, (self.n_samples if end is None else end))
        self.x_type = x_type
        self.y_type = y_type
        self.index = None
        
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
            self.__get_src_impl = self.__get_src_pandas
            self.iter = [i for i in range(0, len(self.index), self.batch_size)]
            
        elif isinstance(X, np.ndarray):
            if isinstance(y, pd.Series):
                self.y = y.to_numpy(y_type)
            elif not (y is None or isinstance(y, np.ndarray)):
                raise Exception('The `y` should be either a `pandas.Series` or `numpy.ndarray` or `None`')
                
            self.__get_item_impl = self.__get_item_numpy
            self.__get_src_impl = self.__get_src_numpy
            self.iter = [i for i in range(self.start_idx, self.end_idx, self.batch_size)]
        else:
            raise Exception('Unsupported `X` type')
            
        self.cnt = len(self.iter)
        if (shuffle):
            rng = np.random.default_rng(random_seed)
            rng.shuffle(self.iter)
            
    
    def get_index(self):
        '''
        Returns the index of pandas.Dataframe's slice used for betches. 
        Returns `None` if features are not in `pandas.DataFrame`
        '''
        return self.index
    

    def get_X(self):
        '''
        Returns features (filtered and sliced if applicable)
        '''
        return self.__get_src_impl(self.X)
    
        
    def get_y(self):
        '''
        Returns target
        '''
        return self.__get_src_impl(self.y)
    
    
    def get_shape(self):
        '''
        Returns input shape of the single window
        '''
        return (self.window_size, self.n_features)
        

    def __len__(self):
        '''
        Returns number of batches
        '''
        return self.cnt
    
    
    def __getitem__(self, idx):
        '''
        Returns batch with the `idx`-th index
        '''    
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
    

    def __get_src_pandas(self, src):
        return src.loc[self.index]
    
    
    def __get_src_numpy(self, src):
        return src[self.start_idx:self.end_idx]

    
class MergeWindowGenerator(Sequence):
    '''
    Uses several instances of the `WindowGenerator` class for [randomly] iterating through them
    '''
    
    def __init__(self, items, shuffle=False, random_seed=None):
        '''
        Constructor
        
            Parameters:
                items (iterable(WindowGenerator)): items to iterate through
                shuffle (bool): shuffle or not 
                random_seed (int|None): random state for random numbers generator
                    applicable only if `shuffle` is `True`
                
            Returns:
                instance of the class
        '''
        if not isinstance(items, (list, tuple, np.ndarray)):
            raise Exception('`items` should be iterable')
    
        self.items = items
        self.iter = np.concatenate([[(i, j) for j in range(len(item))] for i, item in enumerate(items)])
        if (shuffle):
            rng = np.random.default_rng(random_seed)
            rng.shuffle(self.iter)
            
        
    def __len__(self):
        '''
        Returns number of batches
        '''
        return len(self.iter)
    
    
    def __getitem__(self, idx):
        '''
        Returns batch with the `idx`-th index
        '''
        i, j = self.iter[idx]
        return (self.items[i])[j]
    
    
    
def create_gens(X, y=None, test_frac=0.1, val_frac=0.1, **kwargs):
    '''
    Creates a a tuple of generators from single dataset
        
        Parameters:
            X (pandas.DataFrame|pandas.Series|numpy.ndarray): features
            y (pandas.Series|numpy.ndarray|str|None): target
            test_frac (float|None): the fraction of the dataset used for testing
            val_frac (float): the fraction of the dataset used for validation
            
        Returns:
            train, validation and test generators
    '''
    sz = X.shape[0]
    test_size = 0 if ((test_frac is None) or (test_frac == 0)) else int(sz * test_frac)
    val_size = int(sz * val_frac)
    train_size = sz - val_size - test_size
    
    train_gen = WindowGenerator(X, y, end=train_size, **kwargs)
    val_gen = WindowGenerator(X, y, start=train_size, end=(train_size + val_size), **kwargs)
    test_gen = None if test_size == 0 else WindowGenerator(X, y, start=(train_size + val_size), **kwargs)
    
    return train_gen, val_gen, test_gen


def create_merge_gens(X, y=None, test_frac=0.1, val_frac=0.1, **kwargs):
    '''
    Creates a a tuple of generators from list of datasets
        
        Parameters:
            X (iterable(...)): list of features
            y (iterable(...)|str|None): list of target or target column name or None
            test_frac (float|None): the fraction of the dataset used for testing
            val_frac (float): the fraction of the dataset used for validation
            
        Returns:
            train, validation and test generators
    '''
    train_gens = list()
    val_gens = list()
    test_gens = list()
    for i in range(len(X)):
        X_ = X[i]
        y_ = (X_[y]) if isinstance(y, str) else (None if y is None else y[i])
        train_gen, val_gen, test_gen = create_gens(X_, y_, test_frac, val_frac, **kwargs)
        train_gens.append(train_gen)
        val_gens.append(val_gen)
        test_gens.append(test_gen)
    
    shuffle = kwargs['shuffle'] if 'shuffle' in kwargs else False
    random_seed = kwargs['random_seed'] if 'random_seed' in kwargs else None
    
    train_gen = MergeWindowGenerator(train_gens, shuffle, random_seed)
    val_gen = MergeWindowGenerator(val_gens, shuffle, random_seed)
    test_gen = MergeWindowGenerator(test_gens, shuffle, random_seed)
    
    return train_gen, val_gen, test_gen
