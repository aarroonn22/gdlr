import misc.utils as utils

import numpy as np

def _rmse(y, y_hat):
    '''
    Calculating root mean squared error score for actual and predicted values.
    '''
    utils._check_dimensions(y, y_hat)
    return np.sqrt(np.square(y - y_hat).sum() / y.shape[0])

def _mse(y, y_hat):
    '''
    Calculating mean squared error score for actual and predicted values.
    '''
    utils._check_dimensions(y, y_hat)
    return np.square(y - y_hat).sum() / y.shape[0]

def _mae(y, y_hat):
    '''
    Calculating absolute mean error score for actual and predicted values.
    '''
    utils._check_dimensions(y, y_hat)
    return abs(y - y_hat).sum() / y.shape[0]

def _r2(y, y_hat):
    '''
    Calculating R-squared score for actual and predicted values.
    '''
    utils._check_dimensions(y, y_hat)
    return 1 - np.square(y - y_hat).sum() / np.square(y - y.mean()).sum()
