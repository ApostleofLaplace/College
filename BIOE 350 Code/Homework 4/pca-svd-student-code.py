import numpy as np 

def center(X,stype='col'):
    '''
    This just does centering, with the same tricks I'm using for standardize().
    '''
    if stype == 'col':
        return (X - X.mean(axis=0)[np.newaxis,:])
    return (X - X.mean(axis=1)[:,np.newaxis])


def standardize(X,stype='col'):
    ''' 
    This standardizes X on either rows or columns (i.e. each row or column 
    has its mean subtracted and is then divided by its standard deviation)
    '''
    if stype == 'col':
        return (X - X.mean(axis=0)[np.newaxis,:])/X.std(axis=0)[np.newaxis,:]
    return (X - X.mean(axis=1)[:,np.newaxis])/X.std(axis=1)[:,np.newaxis]