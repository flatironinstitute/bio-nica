#---------------------------
# Imports
#---------------------------

import numpy as np
import matplotlib.pyplot as plt
import time

from bio_nica import bio_nica
from nonnegative_pca import nonnegative_pca
from two_layer_nsm import two_layer_nsm
from util import permutation_error, add_fill_lines

#---------------------------
# Load dataset
#---------------------------

# Uncomment the desired dataset

# dataset = '3-dim_synthetic'
dataset = '10-dim_synthetic'

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

bionica_err = np.zeros((trials,samples))
nsm_err = np.zeros((trials,samples))
npca_err = np.zeros((trials,samples))

for i_trial in range(trials):
    
    print(f'Trial {i_trial+1}:')
    
    # Shuffle data

    idx = np.random.permutation(samples)
    S_permuted = S[:,idx[:]]
    X_permuted = X[:,idx[:]]

    #---------------------------
    # Bio-NICA
    #---------------------------

    print('Testing Bio-NICA...')

    bionica = bio_nica(s_dim, x_dim, dataset)
    bionica_Y = np.zeros((s_dim,samples))

    start_time = time.time()

    for i_sample in range(samples):
        x = X_permuted[:,i_sample]
        bionica_Y[:,i_sample] = bionica.fit_next(x)

        if i_sample<=100:
            for j in range(s_dim):
                if sum(bionica_Y[0:i_sample,j])==0:
                    bionica.flip_weights(j)

    elapsed_time = time.time() - start_time

    print(f'Elapsed time: {elapsed_time} seconds')

    bionica_err[i_trial,:] = permutation_error(S_permuted,bionica_Y)

    print(f'Final permutation error: {bionica_err[i_trial,-1]}')

    #---------------------------
    # 2-layer NSM
    #---------------------------

    print('Testing 2-layer NSM...')

    nsm_Y = np.zeros((s_dim,samples))
    nsm = two_layer_nsm(s_dim, x_dim, dataset)

    start_time = time.time()

    for i_sample in range(samples):
        x = X_permuted[:,i_sample]
        nsm_Y[:,i_sample] = nsm.fit_next(x)

        if i_sample<=100:
            for j in range(s_dim):
                if sum(nsm_Y[0:i_sample,j])==0:
                    nsm.flip_weights(j)

    elapsed_time = time.time() - start_time

    print(f'Elapsed time: {elapsed_time} seconds')

    nsm_err[i_trial,:] = permutation_error(S_permuted,nsm_Y)

    print(f'Final permutation error: {nsm_err[i_trial,-1]}')

    #---------------------------
    # Nonnegative PCA
    #---------------------------

    print('Testing Nonnegative PCA...')

    npca = nonnegative_pca(s_dim, x_dim, dataset)
    npca_Y = np.zeros((s_dim,samples))

    start_time = time.time()

    # Noncentered whitening

    sig, U = np.linalg.eig(np.cov(X_permuted))

    X_white = U@np.diag(1./np.sqrt(sig))@U.T@X_permuted

    for i_sample in range(samples):
        x = X_white[:,i_sample]
        npca_Y[:,i_sample] = npca.fit_next(x)

        if (i_sample+1)%100==0 and i_sample<=1000:
            for j in range(s_dim):
                if npca_Y[0:i_sample,j].sum()==0:
                    npca.flip_weights(j)

    elapsed_time = time.time() - start_time

    print(f'Elapsed time: {elapsed_time} seconds')

    npca_err[i_trial,:] = permutation_error(S_permuted,npca_Y)

    print(f'Final permutation error: {npca_err[i_trial,-1]}')

#---------------------------
# Comparison plots
#---------------------------

print('Plotting...')

linewidth = 3

t = list(range(samples))

fig = plt.figure(figsize=(5,5))

ax = plt.subplot(1, 1, 1)

add_fill_lines(ax, t, bionica_err, plot_kwargs={'ls': '-', 'lw': linewidth, 'label': 'Bio-NICA'})
add_fill_lines(ax, t, nsm_err, plot_kwargs={'ls': '-.', 'lw': linewidth, 'label': '2-layer NSM'})
add_fill_lines(ax, t, npca_err, plot_kwargs={'ls': ':', 'lw': linewidth, 'label': 'Nonnegative PCA'})

ax.loglog()

plt.title(f"{s_dim}-dimensional source data")
plt.grid()
plt.legend(loc = 'lower left')
plt.xlabel('Sample #')
plt.ylabel('Permutation error')
plt.xlim((10,samples))

plt.show()
# plt.savefig(f'plots/{s_dim}-dim_synthetic.png', dpi=300, transparent='true', bbox_inches='tight')