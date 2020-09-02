#!/usr/bin/env python

import argparse
import code # for code.interact(local=dict(globals(), **locals()) ) debugging
import glob
import imageio
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from quadprog import solve_qp
import scipy.linalg
from scipy.optimize import linear_sum_assignment
import time

from typing import Dict, List, Tuple # for mypy

######################
# bionica - comparison of NICA algorithms
#
# Reproduces experiments from the paper:
# "A biologically plausible single-layer network for blind nonnegative source separation"

def main():
	cfg = handle_args()
	y_dim = s_dim
	A, S = load_data(cfg)
    
    s_dim = S.shape[0]
    samples = S.shape[1]
    
    iters = cfg.epochs*samples
    
    S_permuted = np.zeros((trials,iters))
    
    for m in cfg.epochs:
        idx = np.random.permutation(samples)
        S_permuted[:,m*samples,(m+1)*samples]=S[:,idx[:]]
        
    X = A@S_permuted # generate mixtures
    
    error = np.zeros((trials,iters))

    for n in range(trials):

        print('Trial {}:'.format(n+1))

        start_time = time.time()

        if cfg.algorithm == 'bio_nica':
            dispatch_bio_nica(cfg, s_dim, X, a, b) # each algorithm should return a matrix Y of size s_dim by iters
        elif cfg.algorithm == '2nsm':
            dispatch_nsm(cfg, s_dim, X)
        elif cfg.algorithm == 'nn_pca':
            dispatch_nnpca(cfg, s_dim, X)
        elif cfg.algorithm == 'fast_ica':
            dispatch_fastica(cfg, s_dim, X)

        # compute the optimal permutation matrix

        perm = np.round(np.corrcoef(S_permuted,Y))[:s_dim,s_dim:] # if the algorithm performs well, rounding the correlation matrix yields the optimal permutation matrix

        # compute error

        for t in range(1,iters):

            error_t = (norm(S_permuted[n,:,t] - perm@Y[n,:,t])**2)/s_dim
            error[n,t] = error[n,t-1] + (error_t - error[n,t-1])/t

    # save data

    np.save(f"error/{cfg.dataset}/{cfg.algorithm}/error.npy", error)

def handle_args():
	parser = argparse.ArgumentParser(description="This does a thing")
# 	parser.add_argument("--s_dim",	type=int, default=3, help="Source dimension")
	parser.add_argument("--x_dim",	type=int, default=3, help="Mixed Stimuli Dimension, must satisfy x_dim >= s_dim")
	parser.add_argument("--trials",	type=int, default=1, help="What does this do?")
	parser.add_argument("--epochs",	type=int, default=1, help="What does this do?")
	parser.add_argument("--dataset", action='store_true', default='image' help="What does this do?")

	parser.add_argument("--algorithm", required=True, choices=['bio_nica', '2nsm', 'nn_pca', 'fast_ica'], help="Which algorithm to run. Valid: bio_nica, 2nsm, nn_pca, fast_ica")

	ret = parser.parse_args()
	return ret

######################
# Dispatch functions. Each of these performs an algorithm over the loaded data, saving the error and
# runtime information for later analysis.

def dispatch_bio_nica(cfg, s_dim, iters, X) -> None:
    # initialize a random matrix W whose rows have norm 1:

    W = np.random.randn(y_dim,x_dim)
    for i in range(y_dim):
        W[i,:] = W[i,:]/norm(W[i,:])

    # initialize M to be the identity matrix, choosing M to be a random postive definite matrix can lead to problems with degeneracy:

    M = np.eye(y_dim)

    # inialize running means:

    x_bar = 0
    c_bar = 0

    start_time = time.time()

    for t in range(iters):

        # compute step size:

        eta_t = 1/(a+b*t)

        x = X[:,i]
        x_bar = x_bar + (x - x_bar)/(t+1) # running average of x 
        x_hat = x - x_bar # centered x

        c = W@x
        c_bar = c_bar + (c - c_bar)/100 # running weighted average of c
        c_hat = c - c_bar # centered c

        # neural dynamics

        y = solve_qp(M, c, np.eye(s_dim), np.zeros(s_dim))[0]
        Y[n,t,:] = y

        # synaptic updates

        W = W + 2*eta_t*(np.outer(y,x) - np.outer(c_hat,x_hat))
        M = M + (eta_t/tau)*(np.outer(y,y) - M)

        # the matrix M can sometimes exhibit instaiblities due to noise, which can be addressed with the following ad hoc fix (the alternative is to reduce the learning rate):

        if scipy.linalg.det(M)<.001:
            M = M + .1*np.eye(s_dim)
            print('M close to degenerate')

        # check to see if any neurons (i.e. components of y) are not firing after the first 10 samples

        if t==10:
            for j in range(s_dim):
                if sum(Y[n,0:t+1,j])==0:
                    W[j,:] = -W[j,:]
                    print('flip',j)

	pass

def dispatch_nsm(cfg, s_dim, X) -> None:
	pass

def dispatch_nnpca(cfg, s_dim, X) -> None:
	pass

def dispatch_fastica(cfg, s_dim, X) -> None:
	pass

######################

def load_data(cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	""" Loads the datasets for the comparison; source path depends on the --image_Data flag """
	if cfg.image_data:
		path_prefix = os.path.join(['data', 'images'])
	else:
		path_prefix = os.path.join(['data', f'{cfg.s_dim}-dim-synthetic'])
	A = np.load(os.path.join([path_prefix, 'mixing-matrix.npy']))
	S = np.load(os.path.join([path_prefix, 'sources.npy'      ]))
	return A, S

#####
main()
