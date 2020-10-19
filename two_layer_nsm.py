# Title: two_layer_nsm.py
# Description: Implementation of a 2-layer network for Nonnegative Independent Component Analysis
# Author: David Lipshutz (dlipshutz@flatironinstitute.org)
# Reference: C. Pehlevan and D.B. Chklovskii "Blind nonnegative source separation using biological neural networks" (2017)

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

    return 1.0 / (t + 100)


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

    def __init__(self, s_dim, x_dim, Whx0=None, Wgh0=None, Wyh0=None, Wyy0=None, learning_rate=eta):

        if Whx0 is not None:
            assert Whx0.shape == (s_dim, s_dim), "The shape of the initial guess Whx0 must be (s_dim,x_dim)=(%d,%d)" % (s_dim, x_dim)
            Whx = Whx0
        else:
            Whx = np.random.normal(0, 1.0 / np.sqrt(x_dim), size=(s_dim, x_dim))

        if Wgh0 is not None:
            assert Wgh0.shape == (s_dim, s_dim), "The shape of the initial guess Wgh0 must be (s_dim,s_dim)=(%d,%d)" % (s_dim, s_dim)
            Wgh = Wgh0
        else:
            Wgh = np.eye(s_dim)

        if Wyh0 is not None:
            assert Wyh0.shape == (s_dim, s_dim), "The shape of the initial guess Whx0 must be (s_dim,x_dim)=(%d,%d)" % (s_dim, s_dim)
            Wyh = Wyh0
        else:
            Wyh = np.random.normal(0, 1.0 / np.sqrt(x_dim), size=(s_dim, s_dim))

        if Wyy0 is not None:
            assert Wyy0.shape == (s_dim, s_dim), "The shape of the initial guess Wgh0 must be (s_dim,s_dim)=(%d,%d)" % (s_dim, s_dim)
            Wyy = Wyy0
        else:
            Wyy = np.eye(s_dim)

        self.eta = learning_rate
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
        self.D = np.zeros(s_dim)

    def fit_next(self, x):
        
        assert x.shape == (self.x_dim,)

        t, s_dim, x_bar, h_bar, g_bar, Whx, Wgh, Wyh, Wyy, D  = self.t, self.s_dim, self.x_bar, self.h_bar, self.g_bar, self.Whx, self.Wgh, self.Wyh, self.Wyy, self.D
        
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
        
        error = 1; steps = 0; max_steps = 100; y = np.zeros(s_dim)
        
        while steps<max_steps or error>10**-4:
            
            y_old = y
                        
            for j in range(s_dim):
                y[j] = max(c[j] - Wyy[j,:]@y_old, 0.0)
                
            error = np.linalg.norm(y-y_old)/np.linalg.norm(y_old+0.001)
            
            steps = steps + 1
            
        # synaptic updates
        
        y_sq = y**2
        
        D = np.maximum(10,.9*D + y_sq)

        Wyh = Wyh + np.linalg.inv(np.diag(D))@(np.outer(y,h) - np.diag(y_sq)@Wyh)
        Wyy = Wyy + np.linalg.inv(np.diag(D))@(np.outer(y,y) - np.diag(y_sq)@Wyy)
        
        for j in range(s_dim):
            Wyy[j,j] = 0

        self.Wyh = Wyh
        self.Wyy = Wyy
        self.D = D
        
        self.t += 1
        
        return y
    
    def flip_weights(self,j):
        
        assert 0<=j<self.s_dim
        
        t, Wyh = self.t, self.Wyh
        
        Wyh[j,:] = -Wyh[j,:]
        
        print(f'After iteration {t}, flipped the weights of row {j}')
       
        self.Wyh = Wyh