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

def get_folds(time_range, offset, k_folds,
              test_size=timedelta(), min_test_size=timedelta(), max_test_size=timedelta(weeks=9999),
              train_size=timedelta(), min_train_size=timedelta()):
    
    for parameter in [offset, test_size, min_test_size, max_test_size, train_size, min_train_size]:
        if not(isinstance(parameter, timedelta)) and parameter is not None: 
            raise AttributeError(f'Please, {parameter} must be a timedelta or None.')
        
    _max = max(time_range)
    _min = min(time_range)    
    test_size_type = 'max' if test_size == timedelta() else 'fixed'
    train_size_type = 'max' if train_size == timedelta() else 'fixed'

    stop_splitting = False
    k_fold = 1
    folds = {}
    while not(stop_splitting):
            
        if test_size_type=='fixed':
            end = _max - (k_fold - 1) * offset           
            split = _max - (k_fold - 1) * offset - max(offset, test_size)
        else:
            end = _max
            split = _max - (k_fold - 1) * offset - max(offset, min_test_size)
        if train_size_type=='fixed':
            begin = split - train_size
        else:
            begin = _min

        test_range = end - split
        train_range = split - begin      
        
        if ((k_fold > k_folds) or 
            (train_range <= timedelta()) or 
            (begin < _min) or
            ((test_size_type=='max') and (test_range > max_test_size)) or
            ((train_size_type=='max') and (train_range < min_train_size))):
            stop_splitting = True
            continue
        folds[k_fold] = {'begin': begin,
                        'split': split,
                        'end': end}
        k_fold += 1
    return folds

def view_folds_graph(folds, _format='%m-%d-%y', figsize=(16, 6)):
    plt.figure(figsize=figsize)
    plt.xticks(rotation=90)
    for k_fold in folds:
        plt.text(folds[k_fold]['begin'], k_fold + 0.05, folds[k_fold]['begin'].strftime(format=_format), ha='center')
        plt.hlines(k_fold, folds[k_fold]['begin'], folds[k_fold]['split'], 'b')
        plt.text(folds[k_fold]['split'], k_fold + 0.05, folds[k_fold]['split'].strftime(format=_format), ha='center')
        plt.hlines(k_fold, folds[k_fold]['split'], folds[k_fold]['end'], 'r')
        plt.text(folds[k_fold]['end'], k_fold + 0.05, folds[k_fold]['end'].strftime(format=_format), ha='center')
    return None

def view_folds_frame(folds):
    return pd.DataFrame(folds).T

def validate(folds, X, y, time_series, model, *metric_functions, verbose=True):
     #= folds.copy()
    results = {}
    for k_fold, fold in folds.items():
        results[k_fold] = {}
        if verbose: last_time = datetime.now()
        X_train, X_test, y_train, y_test = train_test_split(X, y, time_series,
                                                            fold['begin'], 
                                                            fold['split'],
                                                            fold['end'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        for metric_function in metric_functions:
            results[k_fold][metric_function.__name__] = metric_function(y_test, y_pred)
        if verbose: 
            finish_estimate = (len(folds) - k_fold) * (datetime.now() - last_time) + datetime.now() 
            finish_estimate = str(finish_estimate).split('.')[0]
            clear_output(wait=True)
            print(f'({k_fold + 1} of {len(folds)}) Finish estimated to: {finish_estimate}')
    if verbose: 
        clear_output(wait=True)
        print("Multi-period validation completed!")             
    return(results)