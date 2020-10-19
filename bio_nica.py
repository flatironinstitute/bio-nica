# Title: bio_nica.py
# Description: Implementation of Bio-NICA, a single layer network for Nonnegative Independent Component Analysis.
# Author: David Lipshutz (dlipshutz@flatironinstitute.org)
# Reference: D. Lipshutz and D. B. Chklovskii, “Bio-NICA: A biologically inspired single-layer network for Nonnegative Independent Component Analysis"

##############################
# Imports
import numpy as np
from quadprog import solve_qp

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

    return 1.0 / (t + 100)


class bio_NICA:
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

    def __init__(self, s_dim, x_dim, M0=None, W0=None, learning_rate=eta, tau=0.5):

        if M0 is not None:
            assert M0.shape == (s_dim, s_dim), "The shape of the initial guess Minv0 must be (s_dim,s_dim)=(%d,%d)" % (s_dim, s_dim)
            M = M0
        else:
            M = np.eye(K)

        if W0 is not None:
            assert W0.shape == (s_dim, x_dim), "The shape of the initial guess W0 must be (s_dim,x_dim)=(%d,%d)" % (s_dim, x_dim)
            W = W0
        else:
            W = np.random.normal(0, 1.0 / np.sqrt(x_dim), size=(s_dim, x_dim))

        self.eta = learning_rate
        self.t = 0

        self.s_dim = s_dim
        self.x_dim = x_dim
        self.tau = tau
        self.x_bar = np.zeros((x_dim,1))
        self.c_bar = np.zeros((s_dim,1))
        self.M = M
        self.W = W

    def fit_next(self, x):

        assert x.shape == (self.x_dim,1)

        t, s_dim, tau, x_bar, c_bar, W, M  = self.t, self.s_dim, self.tau, self.x_bar, self.c_bar, self.W, self.M
        
        # project inputs
        
        c = W@x
        
        # neural dynamics

        y = solve_qp(M, c, np.eye(s_dim), np.zeros(s_dim))[0]

        # update running averages (bar) and center (hat)

        x_bar += (x - x_bar)/(t+1) 
        x_hat = x - x_bar

        c_bar += (c - c_bar)/(t+1)
        c_hat = c - c_bar
        
        self.x_bar = x_bar
        self.c_bar = c_bar

        # synaptic updates
        
        step = self.eta(t)

        W += 2*step*(np.outer(y,x) - np.outer(c_hat,x_hat))
        M += (step/tau)*(np.outer(y,y) - M)
        
        self.M = M
        self.W = W
        
        self.t += 1
        
        return y