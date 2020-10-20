#---------------------------
# Imports
#---------------------------

import numpy as np
import matplotlib.pyplot as plt
import math
import time

from bio_nica import bio_nica
from nonnegative_pca import nonnegative_pca
from two_layer_nsm import two_layer_nsm
from util import permutation_error

#---------------------------
# Load dataset
#---------------------------

# Uncomment the desired dataset

dataset = '3-dim_synthetic'
# dataset = '10-dim_synthetic'

print(f'Loading dataset...')

S = np.load(f'datasets/{dataset}/sources.npy')
X = np.load(f'datasets/{dataset}/mixtures.npy')

s_dim = S.shape[0]
x_dim = X.shape[0]
samples = S.shape[1]

#---------------------------
# Parameters
#---------------------------

trials = 10
        
#---------------------------
# Bio-NICA
#---------------------------

print('Testing Bio-NICA...')

bionica_err = np.zeros((trials,samples))

for i_trial in range(trials):
    
    print(f'Trial {i_trial+1}:')

    bionica = bio_nica(s_dim, x_dim, dataset)
    bionica_Y = np.zeros((s_dim,samples))

    start_time = time.time()

    for i_sample in range(samples):
        x = X[:,i_sample]
        bionica_Y[:,i_sample] = bionica.fit_next(x)

        if i_sample<=100:
            for j in range(s_dim):
                if sum(bionica_Y[0:i_sample,j])==0:
                    bionica.flip_weights(j)

    elapsed_time = time.time() - start_time

    print(f'Elapsed time: {elapsed_time} seconds')

    bionica_err[i_trial,:] = permutation_error(S,bionica_Y)
    
    print(f'Final permutation error: {bionica_err[i_trial,-1]}')

print('Bio-NICA test complete!')

#---------------------------
# 2-layer NSM
#---------------------------

# print('Testing 2-layer NSM...')

# nsm_Y = np.zeros((s_dim,samples))
# nsm = two_layer_nsm(s_dim, x_dim, dataset)

# start_time = time.time()
    
# for i_sample in range(samples):
#     x = X[:,i_sample]
#     nsm_Y[:,i_sample] = nsm.fit_next(x)

#     if i<=100:
#         for j in range(s_dim):
#             if sum(nsm_Y[0:i_sample,j])==0:
#                 nsm.flip_weights(j)
    
# elapsed_time = time.time() - start_time

# print(f'Elapsed time: {elapsed_time} seconds')

# nsm_err = permutation_error(S,nsm_Y)

# print('2-layer NSM test complete!')

#---------------------------
# Nonnegative PCA
#---------------------------

# print('Testing Nonnegative PCA...')

# npca_Y = np.zeros((s_dim,samples))
# npca = nonnegative_pca(s_dim, x_dim, dataset)

# start_time = time.time()

# # Noncentered whitening

# sig, U = np.linalg.eig(np.cov(X))

# X_white = U@np.diag(1./np.sqrt(sig))@U.T@X
    
# for i_sample in range(samples):
#     x = X_white[:,i_sample]
#     npca_Y[:,i_sample] = npca.fit_next(x)

#     if i<=100:
#         for j in range(s_dim):
#             if sum(npca_Y[0:i_sample,j])==0:
#                 npca.flip_weights(j)
    
# elapsed_time = time.time() - start_time

# print(f'Elapsed time: {elapsed_time} seconds')

# npca_err = permutation_error(S,npca_Y)

# print('Nonnegative PCA test complete!')

print('Plotting...')

#---------------------------
# Comparison plots
#---------------------------

fig = plt.figure(figsize=(8,8))

plt.loglog(bionica_err[0,:], lw=3, ls="-", label="Bio-NICA")
# plt.loglog(nsm_err, lw=3, ls="-:", label="2-layer NSM")
# plt.loglog(npca_err, lw=3, ls=":", label="Nonnegative PCA")
plt.grid()
plt.legend()
plt.xlabel('Sample #')
plt.ylabel('Permutation error')
plt.xlim((100,samples))

plt.show()