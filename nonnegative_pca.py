# Title: nonnegative_pca.py
# Description: Implementation of the Nonnegative PCA algorithm for Nonnegative Independent Component Analysis
# Author: David Lipshutz (dlipshutz@flatironinstitute.org)
# Reference: M.D. Plumbley and E. Oja "A ‘Nonnegative PCA’ algorithm for independent component analysis" (2004)

##############################
# Imports
import numpy as np

##############################


def eta(t):
    """
    Parameters:
    ====================
    t -- time at which learning rate is to be evaluated
    
    Output:
    ====================
    step -- learning rate at time t
    """

    return 1.0 / (1000 + .01*t)


class nonnegative_pca:
    """
    Parameters:
    ====================
    s_dim         -- Dimension of sources
    x_dim         -- Dimension of mixtures
    W0            -- Initial guess for the forward weight matrix W, must be of size s_dim by x_dim
    learning_rate -- Learning rate as a function of t
    
    Methods:
    ========
    fit_next()
    """

    def __init__(self, s_dim, x_dim, W0=None, learning_rate=eta):

        if W0 is not None:
            assert W0.shape == (s_dim, x_dim), "The shape of the initial guess W0 must be (s_dim,x_dim)=(%d,%d)" % (s_dim, x_dim)
            W = W0
        else:
            W = np.random.normal(0, 1.0 / np.sqrt(x_dim), size=(s_dim, x_dim))

        self.eta = learning_rate
        self.t = 0

        self.s_dim = s_dim
        self.x_dim = x_dim

        self.W = W

    def fit_next(self, x):
        
        assert x.shape == (self.x_dim,)

        t, s_dim, W = self.t, self.s_dim, self.W
        
        # project inputs
        
        y = np.maximum(W@x,0)

        # synaptic updates
        
        step = self.eta(t)

        W += step*(np.outer(y,x)-np.outer(y,y)@W)

        self.W = W
        
        self.t += 1
        
        return y
    
    def flip_weights(self,j):
        
        assert 0<=j<self.s_dim
        
        t, W = self.t, self.W
        
        W[j,:] = -W[j,:]
        
        print(f'After iteration {t}, flipped the weights of row {j}')
       
        self.W = W