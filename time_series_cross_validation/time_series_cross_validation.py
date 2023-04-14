import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from IPython.display import clear_output

def train_test_split(X, y, time_series, begin, split, end):
    X_train = X[(time_series >= begin) & (time_series < split)]
    X_test = X[(time_series >= split) & (time_series <= end)]
    y_train = y[(time_series >= begin) & (time_series < split)]
    y_test = y[(time_series >= split) & (time_series <= end)]
    return X_train, X_test, y_train, y_test

class Time_series_CV:
    def __init__(self, time_range, offset, k_folds=6,
                 test_size=None, min_test_size=None, max_test_size=None,
                 train_size=None, min_train_size=None):
        
        for parameter in [offset, test_size, min_test_size, max_test_size, train_size, min_train_size]:
            if not(isinstance(parameter, timedelta)) and parameter is not None: 
                raise AttributeError('Please, argument must be a timedelta or None.')
                
        self.__max = max(time_range)
        self.__min = min(time_range)
        self.__offset = offset
        self.__k_folds = k_folds
        
        self.__train_size_type = 'max' if train_size is None else 'fixed'
        self.__test_size_type = 'max' if test_size is None else 'fixed' 
        
        self.__train_size = timedelta() if train_size is None else train_size
        self.__min_train_size = timedelta() if min_train_size is None else min_train_size
        
        self.__test_size = timedelta() if test_size is None else test_size
        self.__min_test_size = timedelta() if min_test_size is None else min_test_size
        self.__max_test_size = timedelta(weeks=9999) if max_test_size is None else max_test_size
        
        self.folds = self.__get_folds()
        
        return None
    
    def __get_folds(self):
        stop_splitting = False
        k_fold = 1
        folds = {}
        while not(stop_splitting):
            
            if self.__test_size_type=='fixed':
                end = self.__max - (k_fold - 1) * self.__offset            
                split = self.__max - (k_fold - 1) * self.__offset - max(self.__offset, self.__test_size)
            else:
                end = self.__max
                split = self.__max - (k_fold - 1) * self.__offset - max(self.__offset, self.__min_test_size)
            if self.__train_size_type=='fixed':
                begin = split - self.__train_size
            else:
                begin = self.__min
            
            test_range = end - split
            train_range = split - begin
            
            if ((k_fold > self.__k_folds) or 
                (train_range <= timedelta()) or 
                (begin < self.__min) or
                ((self.__test_size_type=='max') and (test_range > self.__max_test_size)) or
                ((self.__train_size_type=='max') and (train_range < self.__min_train_size))):
                stop_splitting = True
                continue
            folds[k_fold] = {'begin': begin,
                           'split': split,
                           'end': end}
            k_fold += 1
        return folds
    
    def view_folds_graph(self, _format='%m-%d-%y'):
        plt.figure(figsize=(16, 6))
        plt.xticks(rotation=90)
        df = pd.DataFrame(self.folds).T
        for k_fold in df.index:
            plt.text(df.loc[k_fold, 'begin'], k_fold + 0.05, df.loc[k_fold, 'begin'].strftime(format=_format), ha='center')
            plt.hlines(k_fold, df.loc[k_fold, 'begin'], df.loc[k_fold, 'split'], 'b')
            plt.text(df.loc[k_fold, 'split'], k_fold + 0.05, df.loc[k_fold, 'split'].strftime(format=_format), ha='center')
            plt.hlines(k_fold, df.loc[k_fold, 'split'], df.loc[k_fold, 'end'], 'r')
            plt.text(df.loc[k_fold, 'end'], k_fold + 0.05, df.loc[k_fold, 'end'].strftime(format=_format), ha='center')
        return None
    
    def view_folds(self, _format='%m-%d-%y'):
        return pd.DataFrame(self.folds).T
            
    def validate(self, X, y, time_series, model=None, metrics_function=None):
        print("Running first k-fold...")
        for k_fold, fold in self.folds.items():
            last_time = datetime.now()
            X_train, X_test, y_train, y_test = train_test_split(X, y, time_series,
                                                                fold['begin'], 
                                                                fold['split'],
                                                                fold['end'])
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = metrics_function(y_test, y_pred)
            for metric, score in metrics.items():
                fold[metric] = score

            finish_estimate = (len(self.folds) - k_fold) * (datetime.now() - last_time) + datetime.now() 
            finish_estimate = str(finish_estimate).split('.')[0]
            clear_output(wait=True)
            print(f'({k_fold + 1} of {len(self.folds)}) Finish estimated to: {finish_estimate}')

        clear_output(wait=True)
        print("Multi-period validation completed!")             
            
        return None