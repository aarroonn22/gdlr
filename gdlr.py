# -*- coding: utf-8 -*-

import misc.metrics as metrics
from misc.metrics import *

import misc.utils as utils
from misc.utils import *

import numpy as np
import matplotlib.pyplot as plt


class grad_desc_lin_reg:
    '''
    Algorithm to fit a linear regression line using gradient descent.
    '''
    def __init__(self):
        ## initialising the intercept, coefficients and errors parameters
        self.intercept = None
        self.coefficients = None
        self.errors = np.empty(0)
        
        
    def fit(self, X, y, learning_rate = 0.001, epochs = 1, best_epoch = False):
        '''
        Fits a regression line on X predicting y based on gradient descent.
        
        Parameters:
        - X: predictors. two dimensional numpy.ndarray type.
        - y: dependent value. scalar numpy.ndarray type.
        - learning_rate: controls the penalty function of the gradient descent. numeric type. default value: 0.001.
        - epochs: sets the number of epochs the model is fitted. integer type. default value: 1.
        - best_epoch: whether the number of best performing epoch based on errors should be returned.
          possible values: [False, 'rmse', 'mse', 'mae']. if False: no epoch is returned. if 'rmse', the number of
          the epoch with the lowest root mean squared error score is returned. if 'mse', the number of the epoch with
          the lowest mean square error is returned. if 'mae', the number of the eopch with the lowest mean absolute
          error score is returned. default value: False.
        '''
        ## checking whether the dimensions of input variables match
        utils._check_dimensions(X, y)
        
        ## setting initial weights and errors
        w_b0 = 0
        w_bn = np.zeros(X.shape[1])
        errs = np.empty(0)
        ## preserving the length of the predicted for later reference
        self.d_out = y.shape[0]
        
        ## calculating the intercept and coefficient values and error scores
        for n in range(epochs): # going through n epochs
            for m in range(y.shape[0]): # for m data points
                ## we calculate the error based on current weights
                error = (w_b0 + sum([(w_bn[i] * X[m, i]) for i in range(X.shape[1])])) - y[m]
                ## we add the current error score to the error list (errs)
                errs = np.append(errs, error)
                ## we update the weight for the intercept
                w_b0 = (w_b0 - learning_rate * error)
                ## for each predictor, we update the coefficient
                for x in range(X.shape[1]):
                    w_bn[x] = (w_bn[x] - learning_rate * error * X[m][x])
                    
                    
        ## setting the class attributes to the model's values
        self.intercept = w_b0
        self.coefficients = w_bn
        self.errors = errs
        
        ## calculating error scores for epochs based on given parameters
        if best_epoch == 'rmse':
            score = [metrics._rmse(self.errors[n: n + self.d_out], np.zeros(self.d_out))
                     for n in range(0, self.errors.shape[0], self.d_out)]
            self.b_e = score.index(min(score)) + 1
            print(f'Bets epoch is no.: {self.b_e}')
            
        elif best_epoch == 'mse':
            score = [metrics._mse(self.errors[n: n + self.d_out], np.zeros(self.d_out))
                     for n in range(0, self.errors.shape[0], self.d_out)]
            self.b_e = score.index(min(score)) + 1
            print(f'Bets epoch is no.: {self.b_e}')
            
        elif best_epoch == 'mae':
            score = [metrics._mae(self.errors[n: n + self.d_out], np.zeros(self.d_out))
                     for n in range(0, self.errors.shape[0], self.d_out)]
            self.b_e = score.index(min(score)) + 1
            print(f'Bets epoch is no.: {self.b_e}')
        
        else:
            pass
        
    
    def predict(self, X):
        '''
        Makes predictions on input data using the current model fit.
        
        Parameters:
        - X: predictor values. two dimensional numpy.ndarray type.
        
        Returns:
        - predicted dependent values. scalar numpy.ndarray type.
        '''
        return np.array([(self.intercept + sum([X[m, n] * self.coefficients[n]
                                                for n in range(X.shape[1])]))
                         for m in range(X.shape[0])])
    
    
    
    def rmse(self, y, X):
        '''
        Returns root mean squared error of inputs based on the current fit.
        
        Parameters:
        - y: values of the dependent variable. scalar numpy.ndarray type.
        - X: values of the predictor. two dimensional numpy.ndarray type.
        
        Returns:
        - root mean squared error score for the model on the data. float type.
        '''
        return metrics._rmse(y, self.predict(X))
    
    
    def mse(self, y, X):
        '''
        Returns mean squared error of inputs based on the current fit.
        
        Parameters:
        - y: values of the dependent variable. scalar numpy.ndarray type.
        - X: values of the predictor. two dimensional numpy.ndarray type.
        
        Returns:
        - mean squared error score for the model on the data. float type.
        '''
        return metrics._mse(y, self.predict(X))
        
    
    def mae(self, y, X):
        '''
        Returns mean absolute error of inputs based on the current fit.
        
        Parameters:
        - y: values of the dependent variable. scalar numpy.ndarray type.
        - X: values of the predictor. two dimensional numpy.ndarray type.
        
        Returns:
        - mean absolute error score for the model on the data. float type.
        '''
        return metrics._mae(y, self.predict(X))
    
    
    def r2(self, y, X):
        '''
        Returns R-squared score based on the current fit.
        
        Parameters:
        - y: values of the dependent variable. scalar numpy.ndarray type.
        - X: values of the predictor. two dimensional numpy.ndarray type.
        
        Returns:
        - R-squared score for the model on the data. float type.
        '''
        return metrics._r2(y, self.predict(X))
    
    def error_graph(self, abs_ = False, agg = False, **kwargs):
        '''
        Plots a matplotlib.pyplot.plot graph for the error scores of the model based on given parameters.
        
        Parameters:
        - abs_: whether the plot should use absolute values of errors. possible values: [False, True]. default value:
          False
        - agg: whether error scores should be aggregated by epochs. possible values: [False, 'rmse', 'mse', 'mae'].
          if False: no aggregation will be carried out. is 'rmse': root mean quared error score will be calculated for
          each epoch. if 'mse': mean quared error score will be calculated for each epoch. is 'mae': absolute mean
          error score will be calculated for each epoch. default value: False.
        - **kwargs: input parameters for matplotlib.pyplot.plot().
        '''
        if agg == 'rmse':
            errors = [metrics._rmse(self.errors[n: n + self.d_out], np.zeros(self.d_out))
                      for n in range(0, self.errors.shape[0], self.d_out)]
            
        elif agg == 'mse':
            errors = [metrics._mse(self.errors[n: n + self.d_out], np.zeros(self.d_out))
                      for n in range(0, self.errors.shape[0], self.d_out)]
            
        elif agg == 'mae':
            errors = [metrics._mae(self.errors[n: n + self.d_out], np.zeros(self.d_out))
                      for n in range(0, self.errors.shape[0], self.d_out)]
            
        else:
            errors = self.errors
        
        if abs_:
            errors = abs(errors)
        
        plt.plot(range(1, len(errors) + 1), errors, **kwargs)