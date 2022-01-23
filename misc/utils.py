import numpy as np

def _check_dimensions(*args):
    '''
    Asserting whether the lengths of inputs match.
    '''
    for i in args:
        _check_array(i)
        try:
            assert args[0].shape[0] == i.shape[0]
        except AssertionError:
            print('The dimensions of input parameters mismatch')
            raise
            
def _check_array(*args):
    '''
    Asserting whether the lengths of inputs are of the numpy.ndarray type.
    '''
    for i in args:
        try:
            assert type(i) == np.ndarray
        except:
            print('Some inputs are not of the numpy.ndarray type')
            raise
