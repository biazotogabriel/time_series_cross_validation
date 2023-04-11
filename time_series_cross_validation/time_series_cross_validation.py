import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import time
from IPython.display import clear_output
import threading
import signal

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
    
    
    class Cronometer(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self, name='chronometer')
            signal.signal(signal.SIGINT, self.signal_handler)
            self.killed = False
            self.total_time = None
            self.k_fold_progress = '0 of 0'
            return None

        def run(self):
            interval = timedelta(seconds=1) 
            while self.total_time is None:
                clear_output(wait=True)
                print('Running first k-fold...')
                time.sleep(1)
            while (not self.killed) and (self.total_time > timedelta()):
                clear_output(wait=True)
                time_to_complete = str(self.total_time).split('.')[0]
                print(f'Time to complete validation: {time_to_complete} ({self.k_fold_progress}).')
                self.total_time -= interval
                time.sleep(1) 
            return None
        
        def update(self, total_time, k_fold_progress):
            self.total_time = total_time
            self.k_fold_progress = k_fold_progress
            return None

        def kill(self):
            self.killed = True
            return None

        def signal_handler(self, sig, frame):
            self.kill()
            self.join()
            print('Interrupted process!')
            return None
    
    def view_folds(self, _format='%m-%d-%y'):
        plt.figure(figsize=(16, 6))
        plt.xticks(rotation=90)
        df = pd.DataFrame(self.folds).T
        for row in range(len(df)):
            plt.text(df.loc[row, 'begin'], row + 0.05, df.loc[row, 'begin'].strftime(format=_format), ha='center')
            plt.hlines(row, df.loc[row, 'begin'], df.loc[row, 'split'], 'b')
            plt.text(df.loc[row, 'split'], row + 0.05, df.loc[row, 'split'].strftime(format=_format), ha='center')
            plt.hlines(row, df.loc[row, 'split'], df.loc[row, 'end'], 'r')
            plt.text(df.loc[row, 'end'], row + 0.05, df.loc[row, 'end'].strftime(format=_format), ha='center')
        return None
            
    def validate(self, X, y, time_series, model=None, metrics_function=None):
        cronometer = self.Cronometer()
        cronometer.start()

        for k_fold, fold in self.folds.items():

            start_time = datetime.now()

            X_train = X[(time_series >= fold['begin']) & (time_series < fold['split'])]
            X_test = X[(time_series >= fold['split']) & (time_series < fold['end'])]
            y_train = y[(time_series >= fold['begin']) & (time_series < fold['split'])]
            y_test = y[(time_series >= fold['split']) & (time_series < fold['end'])]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = metrics_function(y_test, y_pred)
            for metric, score in metrics.items():
                fold[metric] = score

            end_time = datetime.now()
            total_time_missing = (len(self.folds) - k_fold) * (end_time - start_time)
            cronometer.update(total_time_missing, f'{k_fold + 1} of {len(self.folds)}')

        cronometer.kill()
        cronometer.join()
        clear_output(wait=True)
        print("Multi-period validation completed!")    
            
            
        return None