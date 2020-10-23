# Title: nonnegative_pca.py
# Description: Implementation of the Nonnegative PCA algorithm for Nonnegative Independent Component Analysis
# Author: David Lipshutz (dlipshutz@flatironinstitute.org)
# Reference: M.D. Plumbley and E. Oja "A ‘Nonnegative PCA’ algorithm for independent component analysis" (2004)

##############################
# Imports
import numpy as np

##############################

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

    def __init__(self, s_dim, x_dim, dataset=None, W0=None, eta0=0.1, decay=0.1):

        if W0 is not None:
            assert W0.shape == (s_dim, x_dim), "The shape of the initial guess W0 must be (s_dim,x_dim)=(%d,%d)" % (s_dim, x_dim)
            W = W0
        else:
            W = np.random.randn(s_dim,x_dim)
            for i in range(s_dim):
                W[i,:] = W[i,:]/np.linalg.norm(W[i,:])


        # optimal hyperparameters for test datasets
            
        if dataset=='3-dim_synthetic' and s_dim==3 and x_dim ==3:
            eta0 = 0.001
            decay = 0.00001
        elif dataset=='10-dim_synthetic' and s_dim==10 and x_dim==10:
            eta0 = 0.001
            decay = 0.00001

        self.t = 0
        self.eta0 = eta0
        self.decay = decay
        self.s_dim = s_dim
        self.x_dim = x_dim
        self.W = W

    def fit_next(self, x):
        
        assert x.shape == (self.x_dim,)
        
        t, s_dim, W = self.t, self.s_dim, self.W

        # project inputs
        
        y = np.maximum(W@x,0)

        # synaptic updates
        
        step = self.eta0/(1+self.decay*t)

        W += step*(np.outer(y,x)-np.outer(y,y)@W)

        self.W = W
        
        self.t = t+1
        
        return y
    
    def flip_weights(self,j):
        
        t, W = self.t, self.W
                
        W[j,:] = -W[j,:]
                       
        self.W = W