import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder

class ValidateAndRenameColumns(BaseEstimator):
    def __init__(self):
        self.needed_columns = ['age', 'workclass', 'education', 'education_num',
           'marital_status', 'occupation', 'relationship',
           'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']
            
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        self.proper_columns = False
        self._validate(X)
        if not self.proper_type:
            raise TypeError('Data must be formatted as a Dictionary or a pandas DataFrame')
        X = self._reformat(X)
        self._check_columns(X)
        if not self.proper_columns:
            raise ValueError("The following features are missing. {}".format(self.missing_keys))
        
        if self.type == dict:
            X = pd.DataFrame([X], index = range(len([X])))
        data = X.copy()
        
        return data[self.needed_columns]
    
    def _validate(self, X):

        if type(X) == dict:
            self.type = dict
            self.proper_type = True
                
        elif type(X) == pd.core.frame.DataFrame:
            self.type = pd.core.frame.DataFrame
            self.proper_type = True
        else:
            self.proper_type = False    
                
    def _reformat(self, X):
        change_text = lambda x: x.lower().strip().replace(' ', '_').replace('-', '_')
        if self.type == dict:
            for key in X:
                new_key = change_text(key)
                X[new_key] = X.pop(key)
        else:
            X.columns = [change_text(column) for column in X.columns]
            
        return X
    
    def _check_columns(self, X):   
        if self.type == dict:
            if all(key in list(X.keys()) for key in self.needed_columns):
                self.proper_columns = True
                return
            else:
                self.missing_keys = [key for key in self.needed_columns if key not in list(X.keys())]   
        else:
            if all(column in X.columns for column in self.needed_columns):
                self.proper_columns = True
                return
            
            else:
                self.missing_keys = [column for column in self.needed_columns if column not in X.columns]
        return

class HotEncodeMerge():
    def __init__(self):
        
        HotEncodeMerge.encoder = OneHotEncoder(drop="first")
        HotEncodeMerge.columns = ['workclass', 'education', 'education_num','marital_status', 'occupation',
           'relationship', 'native_country']
    
    def fit(self, X, y):
        HotEncodeMerge.encoder.fit(X[HotEncodeMerge.columns])
        return self
    
    def transform(self, X, y=None):
        hot_encoded = HotEncodeMerge.encoder.transform(X[HotEncodeMerge.columns])

        hot_encoded = pd.DataFrame(hot_encoded.todense(), 
                           columns=HotEncodeMerge.encoder.get_feature_names(),
                           index=X.index)
        X = pd.concat([X.drop(HotEncodeMerge.columns, axis = 1), hot_encoded], axis = 1)
        return X
        

class BinAge(BaseEstimator):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def _bin_data(self, x):
        if x < 45 or x > 64:
            return 0 
        else: 
            return 1
    
    def transform(self, X):
        if type(X) == dict:
            X = pd.DataFrame([X], index = len([X]))
        data = X.copy()
        data['age'] = data['age'].apply(self._bin_data)
        
        return data 

class BinCapital(BaseEstimator):
    def __init__(self, col_name):
        self.col_name = col_name
    
    def fit(self, X, y=None):
        return self
    
    def _bin_data(self, x):
        if x > 0:
            return 1
        else:
            return 0
    
    def transform(self, X):
        if type(X) == dict:
            X = pd.DataFrame([X], index = len([X]))
        data = X.copy()
        data[self.col_name] = data[self.col_name].apply(self._bin_data)
        
        return data      