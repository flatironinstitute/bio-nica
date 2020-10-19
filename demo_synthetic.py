import numpy as np
import matplotlib.pyplot as plt
import math
import time

from bio_nica import bio_nica
from nonnegative_pca import nonnegative_pca
from two_layer_nsm import two_layer_nsm
from util import permutation_error

print('Testing NICA algorithms on synthetic data...')

#---------------------------
# Parameters
#---------------------------
# Number of samples, epochs
samples = 10**5
# Source and mixture dimensions
s_dim = 3
x_dim = 3
#---------------------------

#---------------------------
# Generate data
#---------------------------

print(f'Generating {s_dim}-dimensional sparse uniform source data...')

# Generate sparse random samples

U = np.random.uniform(0,math.sqrt(48/5),(s_dim,samples)) # independent non-negative uniform source RVs with variance 1
B = np.random.binomial(1, .5, (s_dim,samples)) # binomial RVs to sparsify the source
S = U*B # sources

print(f'Generating {x_dim}-dimensional mixtures...')

# Generate random mixing matrix and mixtures

A = np.random.randn(x_dim,s_dim) # random mixing matrix
X = A@S
        
#---------------------------
# Run Bio-NICA
#---------------------------

print('Testing Bio-NICA...')

# Set the learning rate

def eta(t):
    return 1/(100+.01*t)

bionica_Y = np.zeros((s_dim,samples))
bionica = bio_nica(s_dim, x_dim, learning_rate=eta, tau=0.1)

start_time = time.time()

for i in range(samples):
    x = X[:,i]
    bionica_Y[:,i] = bionica.fit_next(x)

    if i<=100:
        for j in range(s_dim):
            if sum(bionica_Y[0:i,j])==0:
                bionica.flip_weights(j)
    
elapsed_time = time.time() - start_time

print(f'Elapsed time: {elapsed_time} seconds')

bionica_err = permutation_error(S,bionica_Y)

print('Bio-NICA test complete!')

#---------------------------
# Run 2-layer NSM
#---------------------------

print('Testing 2-layer NSM...')

nsm_Y = np.zeros((s_dim,samples))
nsm = two_layer_nsm(s_dim, x_dim)

start_time = time.time()
    
for i in range(samples):
    x = X[:,i]
    nsm_Y[:,i] = nsm.fit_next(x)

    if i<=100:
        for j in range(s_dim):
            if sum(nsm_Y[0:i,j])==0:
                nsm.flip_weights(j)
    
elapsed_time = time.time() - start_time

print(f'Elapsed time: {elapsed_time} seconds')

nsm_err = permutation_error(S,nsm_Y)

print('2-layer NSM test complete!')

#---------------------------
# Run Nonnegative PCA
#---------------------------

print('Testing Nonnegative PCA...')

npca_Y = np.zeros((s_dim,samples))
npca = nonnegative_pca(s_dim, x_dim)

start_time = time.time()

# Noncentered whitening

sig, U = np.linalg.eig(np.cov(X))

X_white = U@np.diag(1./np.sqrt(sig))@U.T@X
    
for i in range(samples):
    x = X_white[:,i]
    npca_Y[:,i] = npca.fit_next(x)

    if i<=100:
        for j in range(s_dim):
            if sum(npca_Y[0:i,j])==0:
                npca.flip_weights(j)
    
elapsed_time = time.time() - start_time

print(f'Elapsed time: {elapsed_time} seconds')

npca_err = permutation_error(S,npca_Y)

print('Nonnegative PCA test complete!')

print('Plotting...')

#---------------------------
# Comparison plots
#---------------------------

fig = plt.figure(figsize=(8,8))

plt.loglog(bionica_err, lw=3, ls="-", label="Bio-NICA")
plt.loglog(nsm_err, lw=3, ls="-:", label="2-layer NSM")
plt.loglog(npca_err, lw=3, ls=":", label="Nonnegative PCA")
plt.grid()
plt.legend()
plt.xlabel('Sample #')
plt.ylabel('Permutation error')
plt.xlim((100,samples))

plt.show()