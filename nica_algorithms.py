# Title: nica_algorithms.py
# Description: Implementation of online algorithms for Nonnegative Independent Component Analysis, including Bio-NICA, 2-layer NSM and Nonnegative PCA.
# Author: David Lipshutz (dlipshutz@flatironinstitute.org)
# References: D. Lipshutz, C. Pehlevan and D.B. Chklovskii "Bio-NICA: A biologically inspired single-layer network for Nonnegative Independent Component Analysis" (2020)
#             C. Pehlevan, S. Mohan and D.B. Chklovskii "Blind nonnegative source separation using biological neural networks" (2017)
#             M.D. Plumbley and E. Oja "A ‘Nonnegative PCA’ algorithm for independent component analysis" (2004)

##############################
# Imports
import numpy as np
from quadprog import solve_qp

##############################

class bio_nica_indirect:
    """
    Parameters:
    ====================
    s_dim         -- Dimension of sources
    x_dim         -- Dimension of mixtures
    n_dim         -- Dimension of interneurons
    P0            -- Initial guess for the lateral weight matrix P, must be of size s_dim by s_dim
    W0            -- Initial guess for the forward weight matrix W, must be of size s_dim by x_dim
    learning_rate -- Learning rate as a function of t
    tau           -- Learning rate factor for M (multiplier of the W learning rate)
    
    Methods:
    ========
    fit_next()
    flip_weights()
    """

    def __init__(self, s_dim, x_dim, n_dim, dataset=None, P0=None, W0=None, eta0=None, decay=None):

        if P0 is not None:
            assert P0.shape == (s_dim, n_dim), "The shape of the initial guess P0 must be (s_dim,n_dim)=(%d,%d)" % (s_dim, n_dim)
            P = P0
        else:
            P = np.zeros((s_dim,n_dim))
            P[:,:s_dim] = np.eye(s_dim)

        if W0 is not None:
            assert W0.shape == (s_dim, x_dim), "The shape of the initial guess W0 must be (s_dim,x_dim)=(%d,%d)" % (s_dim, x_dim)
            W = W0
        else:
            W = np.random.randn(s_dim,x_dim)
            for i in range(s_dim):
                W[i,:] = W[i,:]/np.linalg.norm(W[i,:])
            
        # optimal hyperparameters for test datasets
            
        if dataset=='3-dim_synthetic':
            if eta0 is None:
                eta0 = 1e-2
            if decay is None:
                decay = 1e-3
        elif dataset=='10-dim_synthetic':
            if eta0 is None:
                eta0 = 1e-2
            if decay is None:
                decay = 1e-3
        elif dataset=='images':
            if eta0 is None:
                eta0 = 1e-2
            if decay is None:
                decay = 1e-4

        self.t = 0
        self.eta0 = eta0
        self.decay = decay
        self.s_dim = s_dim
        self.x_dim = x_dim
        self.n_dim = n_dim
        self.x_bar = np.zeros(x_dim)
        self.y_bar = np.zeros(s_dim)
        self.n_bar = np.zeros(n_dim)
        self.P = P
        self.W = W

    def fit_next(self, x):
        
        assert x.shape == (self.x_dim,)

        t, s_dim, x_dim, x_bar, y_bar, n_bar, W, P  = self.t, self.s_dim, self.x_dim, self.x_bar, self.y_bar, self.n_bar, self.W, self.P
        
        # project inputs
        
        c = W@x
        
        # neural dynamics
        
        y = solve_qp(P@P.T, c, np.eye(s_dim), np.zeros(s_dim))[0]
        n = P.T@y

        # update running means

        x_bar += (x - x_bar)/(t+1)
        y_bar += (y - y_bar)/100
        n_bar += (n - n_bar)/100
        
        self.x_bar = x_bar
        self.y_bar = y_bar
        self.n_bar = n_bar

        # synaptic updates
        
        step = self.eta0/(1+self.decay*t)

        W += step*(np.outer(y-y_bar,x-x_bar) - W)
        P += step*(np.outer(y-y_bar,n-n_bar) - P)
        
        # check to see if P is close to degenerate
        # if so, we add .5*identity and flip the feedforward weights

        if np.linalg.det(P@P.T)<1e-4:
            print('PP.T close to degenerate')
            P[:,:s_dim] += .5*np.eye(s_dim)
            W = -W

        for i in range(s_dim):
            if np.linalg.norm(W[i,:])<1e-4:
                print('New weights')
                W = np.random.randn(s_dim,x_dim)/np.sqrt(x_dim)
                P = np.eye(s_dim)

        self.P = P
        self.W = W
        
        self.t += 1
        
        return y
    
    def flip_weights(self,j):
        
        assert 0<=j<self.s_dim
        
        t, W = self.t, self.W
        
        W[j,:] = -W[j,:]
               
        self.W = W
            
class bio_nica_direct:
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
    flip_weights()
    """

    def __init__(self, s_dim, x_dim, dataset=None, M0=None, W0=None, eta0=None, decay=None, tau=None):

        if M0 is not None:
            assert M0.shape == (s_dim, s_dim), "The shape of the initial guess M must be (s_dim,s_dim)=(%d,%d)" % (s_dim, s_dim)
            M = M0
        else:
            M = np.eye(s_dim)

        if W0 is not None:
            assert W0.shape == (s_dim, x_dim), "The shape of the initial guess W0 must be (s_dim,x_dim)=(%d,%d)" % (s_dim, x_dim)
            W = W0
        else:
            W = np.random.randn(s_dim,x_dim)
            for i in range(s_dim):
                W[i,:] = W[i,:]/np.linalg.norm(W[i,:])
            
        # optimal hyperparameters for test datasets
            
        if dataset=='3-dim_synthetic':
            if eta0 is None:
                eta0 = 1e-1
            if decay is None:
                decay = 1e-2
            if tau is None:
                tau = 0.8
        elif dataset=='10-dim_synthetic':
            if eta0 is None:
                eta0 = 1e-3
            if decay is None:
                decay = 1e-4
            if tau is None:
                tau = .03
        elif dataset=='images':
            if eta0 is None:
                eta0 = 1e-3
            if decay is None:
                decay = 1e-6
            if tau is None:
                tau = .1
        else:
            if eta0 is None:
                eta0 = 0.1
            if decay is None:
                decay = 0.001
            if tau is None:
                tau = 0.5

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
        c_bar += (c - c_bar)/100
        
        self.x_bar = x_bar
        self.c_bar = c_bar

        # synaptic updates
        
        step = self.eta0/(1+self.decay*t)

        W += 2*step*(np.outer(y,x) - np.outer(c-c_bar,x-x_bar))
        M += (step/tau)*(np.outer(y,y) - M)
        
        # check to see if M is close to degenerate
        # if so, we add .1*identity to ensure stability

        if np.linalg.det(M)<10**-4:
#             print('M close to degenerate')
            M += .1*np.eye(s_dim)

        self.M = M
        self.W = W
        
        self.t += 1
        
        return y
    
    def flip_weights(self,j):
        
        assert 0<=j<self.s_dim
        
        t, W = self.t, self.W
        
        W[j,:] = -W[j,:]
               
        self.W = W
        
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
    flip_weights()
    """

    def __init__(self, s_dim, x_dim, dataset=None, Whx0=None, Wgh0=None, Wyh0=None, Wyy0=None, eta0=None, decay=None):

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

        if dataset=='3-dim_synthetic':
            if eta0 is None:
                eta0 = 1e-1
            if decay is None:
                decay = 1e-7
        elif dataset=='10-dim_synthetic':
            if eta0 is None:
                eta0 = 1e-1
            if decay is None:
                decay = 1e-6
        elif dataset=='images':
            if eta0 is None:
                eta0 = 1e-1
            if decay is None:
                decay = 1e-6
        else:
            if eta0 is None:
                eta0 = 0.1
            if decay is None:
                decay = 0.001

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
        
        # check to see if Wyy is close to degenerate
        # if so, we add .1*identity to ensure stability
        
        if np.linalg.det(Wyy)<10**-4:
            Wyy += .1*np.eye(s_dim)


        self.Wyh = Wyh
        self.Wyy = Wyy
        
        self.t += 1
        
        return y
    
    def flip_weights(self,j):
        
        assert 0<=j<self.s_dim
        
        t, Wyh = self.t, self.Wyh
        
        Wyh[j,:] = -Wyh[j,:]
               
        self.Wyh = Wyh
        
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
    flip_weights()
    """

    def __init__(self, s_dim, x_dim, dataset=None, W0=None, eta0=None, decay=None):

        if W0 is not None:
            assert W0.shape == (s_dim, x_dim), "The shape of the initial guess W0 must be (s_dim,x_dim)=(%d,%d)" % (s_dim, x_dim)
            W = W0
        else:
            W = np.random.randn(s_dim,x_dim)
            for i in range(s_dim):
                W[i,:] = W[i,:]/np.linalg.norm(W[i,:])


        # optimal hyperparameters for test datasets
            
        if dataset=='3-dim_synthetic':
            if eta0 is None:
                eta0 = 1e-1
            if decay is None:
                decay = 1e-5
        elif dataset=='10-dim_synthetic':
            if eta0 is None:
                eta0 = 1e-2
            if decay is None:
                decay = 1e-5
        elif dataset=='images':
            if eta0 is None:
                eta0 = 1e-3
            if decay is None:
                decay = 1e-5
        else:
            if eta0 is None:
                eta0 = 1e-2
            if decay is None:
                decay = 1e-3

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