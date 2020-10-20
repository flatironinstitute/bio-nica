# Title: bio_nica.py
# Description: Implementation of Bio-NICA, a single layer network for Nonnegative Independent Component Analysis.
# Author: David Lipshutz (dlipshutz@flatironinstitute.org)

##############################
# Imports
import numpy as np
from quadprog import solve_qp

##############################

class bio_nica:
    """
    Parameters:
    ====================
    s_dim         -- Dimension of sources
    x_dim         -- Dimension of mixtures
    M0            -- Initial guess for the lateral weight matrix M, must be of size s_dim by s_dim
    W0            -- Initial guess for the forward weight matrix W, must be of size s_dim by x_dim
    learning_rate -- Learning rate as a function of t
    tau           -- Learning rate factor for M (multiplier of the W learning rate)
    
    Methods:
    ========
    fit_next()
    """

    def __init__(self, s_dim, x_dim, dataset=None, M0=None, W0=None, eta0=0.01, decay=0.01, tau=0.1):

        if M0 is not None:
            assert M0.shape == (s_dim, s_dim), "The shape of the initial guess M must be (s_dim,s_dim)=(%d,%d)" % (s_dim, s_dim)
            M = M0
        else:
            M = np.eye(s_dim)

        if W0 is not None:
            assert W0.shape == (s_dim, x_dim), "The shape of the initial guess W0 must be (s_dim,x_dim)=(%d,%d)" % (s_dim, x_dim)
            W = W0
        else:
            W = np.random.normal(0, 1.0 / np.sqrt(x_dim), size=(s_dim, x_dim))
            
        # optimal hyperparameters for test datasets
            
        if dataset=='3-dim_synthetic' and s_dim==3 and x_dim ==3:
            eta0 = 0.1
            decay = 0.01
            tau = 0.8
        elif dataset=='10-dim_synthetic' and s_dim==10 and x_dim==10:
            eta0 = 0.001
            decay = 0.0001
            tau = .03
        elif dataset=='image' and s_dim==3 and x_dim==6:
            eta0 = 0.001
            decay = 0.000001
            tau = .05

        self.t = 0
        self.eta0 = eta0
        self.decay = decay
        self.tau = tau
        self.s_dim = s_dim
        self.x_dim = x_dim
        self.x_bar = np.zeros(x_dim)
        self.c_bar = np.zeros(s_dim)
        self.M = M
        self.W = W

    def fit_next(self, x):
        
        assert x.shape == (self.x_dim,)

        t, s_dim, tau, x_bar, c_bar, W, M  = self.t, self.s_dim, self.tau, self.x_bar, self.c_bar, self.W, self.M
        
        # project inputs
        
        c = W@x
        
        # neural dynamics

        y = solve_qp(M, c, np.eye(s_dim), np.zeros(s_dim))[0]

        # update running means

        x_bar += (x - x_bar)/(t+1) 
        c_bar += (c - c_bar)/(t+1)
        
        self.x_bar = x_bar
        self.c_bar = c_bar

        # synaptic updates
        
        step = self.eta0/(1+self.decay*t)

        W += 2*step*(np.outer(y,x) - np.outer(c-c_bar,x-x_bar))
        M += (step/tau)*(np.outer(y,y) - M)

        self.M = M
        self.W = W
        
        self.t += 1
        
        return y
    
    def flip_weights(self,j):
        
        assert 0<=j<self.s_dim
        
        t, W = self.t, self.W
        
        W[j,:] = -W[j,:]
        
#         print(f'After iteration {t}, flipped the weights of row {j}')
       
        self.W = W