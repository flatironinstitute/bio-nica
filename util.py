# Title: util.py
# Description: Various utilities useful for online NICA tests
# Author: David Lipshutz (dlipshutz@flatironinstitute.org)

##############################
# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

##############################

def synthetic_data(s_dim, x_dim, samples):
    """
    Parameters:
    ====================
    s_dim   -- The dimension of sources
    x_dim   -- The dimension of mixtures
    samples -- The number of samples

    Output:
    ====================
    S       -- The source data matrix
    X       -- The mixture data matrix
    """
    
    print(f'Generating {s_dim}-dimensional sparse uniform source data...')

    # Generate sparse random samples

    U = np.random.uniform(0,np.sqrt(48/5),(s_dim,samples)) # independent non-negative uniform source RVs with variance 1
    B = np.random.binomial(1, .5, (s_dim,samples)) # binomial RVs to sparsify the source
    S = U*B # sources

    print(f'Generating {x_dim}-dimensional mixtures...')
    
    # Mixing matrices
    
    if s_dim==3 and x_dim==3:
        A = [[0.0315180, .38793, 0.061132], [-0.78502, 0.165610, 0.124580], [.347820, 0.272950, 0.67793]]
    elif s_dim==10 and x_dim==10:
        A = [[-1.610, .110, .111, .26, -0.01, -1.66, 0.45, 0.48, 0.93, -0.57], 
             [-0.95, -0.05, 0.35, -0.68, 1.14, 0.71, -0.38, -0.20, -0.20, 2.02], 
             [0.54, 2.16, 0.06, -0.08, 0.36, -0.16, -0.22, -1.82, -0.22, 0.40], 
             [-0.98, -0.12, -1.45, -0.58, -0.56, 0.34, -0.51, 0.19, -0.44, -0.15],
             [-0.87, 0.54, 0.68, 1.28, 0.63, 1.04, -0.81, 1.08, -0.65, -0.30], 
             [0.91, 0.84, 0.45, -0.31, -0.14, -1.46, -0.18, 0.48, -0.41, 0.75],
             [-1.20, 1.29, 0.39, -1.40, 0.84, -2.32, -1.54, -0.26, -1.99, -0.34],
             [1.34, 0.75, -1.29, -0.63, -1.63, -1.05, 0.07, 0.09, -0.67, 0.28], 
             [-0.32, -0.38, -0.11, 1.18, -0.41, 0.58, -0.92, 1.09, 0.41, 1.29],
             [2.04, 2.00, -0.50, 0.78, -0.65, -0.93, 0.42, -1.69, -1.16, -0.68]]
    else:
        A = np.random.randn(x_dim,s_dim) # random mixing matrix

    # Generate mixtures
    
    X = A@S
    
    np.save(f'datasets/{x_dim}-dim_synthetic/sources.npy', S)
    np.save(f'datasets/{x_dim}-dim_synthetic/mixtures.npy', X)
    
    
# def image_data(s_dim):


def permutation_error(S, Y):
    """
    Parameters:
    ====================
    S   -- The data matrix of sources
    Y   -- The data matrix of recovered sources
    
    Output:
    ====================
    err -- the (relative) Frobenius norm error
    """
    
    assert S.shape==Y.shape, "The shape of the sources S must equal the shape of the recovered sources Y"

    s_dim = S.shape[0]
    iters = S.shape[1]
    
    err = np.zeros(iters)
    
    # Determine the optimal permutation at the final time point.
    # We solve the linear assignment problem using the linear_sum_assignment package
    
    # Calculate cost matrix:
    
    C = np.zeros((s_dim,s_dim))
    
    for i in range(s_dim):
        for j in range(s_dim):
            for t in range(iters):
                C[i,j] += (S[i,t] - Y[j,t])**2
    
    # Find the optimal assignment for the cost matrix C
    
    row_ind, col_ind = linear_sum_assignment(C)
        
    for t in range(1,iters):

        diff_t = (S[row_ind[:],t] - Y[col_ind[:],t])**2
        error_t = diff_t.sum()/s_dim
        err[t] = err[t-1] + (error_t - err[t-1])/t
    
    return err

def add_fill_lines(axis, t, err, ci_pct=.9, plot_kwargs=None, ci_kwargs=None):
    """
    Parameters:
    ====================
    err -- The data matrix of errors over multiple trials
    """
        
    log_err = np.log(err+10**-6)
    mu = log_err.mean(axis=0)
    sigma = np.std(log_err,axis=0)
    ci_lo, ci_hi = mu - sigma, mu + sigma
    plot_kwargs = plot_kwargs or {}
    ci_kwargs = ci_kwargs or {}
    plot = axis.plot(t, np.exp(mu), **plot_kwargs)
    fill = axis.fill_between(t, np.exp(ci_lo), np.exp(ci_hi), alpha=.1, **ci_kwargs)
    
    return plot, fill