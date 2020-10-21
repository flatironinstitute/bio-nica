# Title: two_layer_nsm.py
# Description: Implementation of a 2-layer network for Nonnegative Independent Component Analysis
# Author: David Lipshutz (dlipshutz@flatironinstitute.org)
# Reference: The algorithm is similar to the algorithm derived in C. Pehlevan and D.B. Chklovskii "Blind nonnegative source separation using biological neural networks" (2017)

##############################
# Imports
import numpy as np
from quadprog import solve_qp

##############################

class two_layer_nsm:
    """
    Parameters:
    ====================
    s_dim         -- Dimension of sources
    x_dim         -- Dimension of mixtures
    Whx           -- Initial guess for the forward weight matrix Whx, must be of size s_dim by x_dim
    Wgh           -- Initial guess for the lateral weight matrix Wgh, must be of size s_dim by s_dim
    Wyh           -- Initial guess for the forward weight matrix Wyh, must be of size s_dim by s_dim
    Wyy           -- Initial guess for the lateral weight matrix Wyy, must be of size s_dim by s_dim
    learning_rate -- Learning rate as a function of t
    
    Methods:
    ========
    fit_next()
    """

    def __init__(self, s_dim, x_dim, dataset=None, Whx0=None, Wgh0=None, Wyh0=None, Wyy0=None, eta0=0.1, decay=0.001):

        if Whx0 is not None:
            assert Whx0.shape == (s_dim, s_dim), "The shape of the initial guess Whx0 must be (s_dim,x_dim)=(%d,%d)" % (s_dim, x_dim)
            Whx = Whx0
        else:
            Whx = np.random.randn(s_dim,x_dim)
            for i in range(s_dim):
                Whx[i,:] = Whx[i,:]/np.linalg.norm(Whx[i,:])

        if Wgh0 is not None:
            assert Wgh0.shape == (s_dim, s_dim), "The shape of the initial guess Wgh0 must be (s_dim,s_dim)=(%d,%d)" % (s_dim, s_dim)
            Wgh = Wgh0
        else:
            Wgh = np.eye(s_dim)

        if Wyh0 is not None:
            assert Wyh0.shape == (s_dim, s_dim), "The shape of the initial guess Whx0 must be (s_dim,x_dim)=(%d,%d)" % (s_dim, s_dim)
            Wyh = Wyh0
        else:
            Wyh = np.random.randn(s_dim,s_dim)
            for i in range(s_dim):
                Wyh[i,:] = Wyh[i,:]/np.linalg.norm(Wyh[i,:])

        if Wyy0 is not None:
            assert Wyy0.shape == (s_dim, s_dim), "The shape of the initial guess Wgh0 must be (s_dim,s_dim)=(%d,%d)" % (s_dim, s_dim)
            Wyy = Wyy0
        else:
            Wyy = np.eye(s_dim)

        if dataset=='3-dim_synthetic' and s_dim==3 and x_dim ==3:
            eta0 = 0.01
            decay = 0.00001
        elif dataset=='10-dim_synthetic' and s_dim==10 and x_dim==10:
            eta0 = 0.01
            decay = 0.00001

        self.eta0 = eta0
        self.decay = decay
        self.t = 0
        self.s_dim = s_dim
        self.x_dim = x_dim
        self.x_bar = np.zeros(x_dim)
        self.h_bar = np.zeros(s_dim)
        self.g_bar = np.zeros(s_dim)
        self.Whx = Whx
        self.Wgh = Wgh
        self.Wyh = Wyh
        self.Wyy = Wyy
    
    def fit_next(self, x):
        
        t, s_dim, x_bar, h_bar, g_bar, Whx, Wgh, Wyh, Wyy  = self.t, self.s_dim, self.x_bar, self.h_bar, self.g_bar, self.Whx, self.Wgh, self.Wyh, self.Wyy
        
        # whitening layer
        
        # neural dynamics
        
        h = np.linalg.inv(Wgh.T@Wgh)@Whx@x
        g = Wgh@h
        
        # update running means

        x_bar = x_bar + (x - x_bar)/(t+1)
        h_bar = h_bar + (h - h_bar)/(t+1)
        g_bar = g_bar + (g - g_bar)/(t+1)
        
        self.x_bar = x_bar
        self.h_bar = h_bar
        self.g_bar = g_bar

        # synaptic updates

        Whx = Whx + (1/(100+t))*(np.outer(h - h_bar,x - x_bar) - Whx)
        Wgh = Wgh + (1/(100+t))*(np.outer(g - g_bar,h - h_bar) - Wgh)
        
        self.Whx = Whx
        self.Wgh = Wgh
        
        # rotation layer
        
        c = Wyh@h
        
        y = solve_qp(Wyy, c, np.eye(s_dim), np.zeros(s_dim))[0]
        
        step = self.eta0/(1+self.decay*t)

        Wyh = Wyh + step*(np.outer(y,h) - Wyh)
        Wyy = Wyy + step*(np.outer(y,y) - Wyy)

        self.Wyh = Wyh
        self.Wyy = Wyy
        
        self.t += 1
        
        return y
    
    def flip_weights(self,j):
        
        assert 0<=j<self.s_dim
        
        t, Wyh = self.t, self.Wyh
        
        Wyh[j,:] = -Wyh[j,:]
               
        self.Wyh = Wyh