def kb_logspace(xmin,xmax,alpha=1.2):
    '''
    Produces a set of logarithmically spaced points in [xmin,xmax].  Points 
    are related as:
            x[i+1] = alpha*x[i]
    Which makes 
            log(x[i+1]) = log(x[i]) + alpha
    (So equal spacing in log means the normal unlogged numbers get further 
    and further apart). numpy.logspace can also do this (feel free to use it 
    instead if you like), but I find the syntax of that function 
    confusing.

    Alpha MUST be > 1. Making alpha smaller (closer to 1) gives you more points, 
    alpha larger gives you fewer.
    '''
    x_logspace = [xmin]
    x_current = alpha*xmin 
    while x_current < xmax:
        x_logspace.append(x_current)
        x_current = x_current*alpha 
    return np.array([int(x) for x in x_logspace])