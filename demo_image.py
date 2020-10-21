#---------------------------
# Imports
#---------------------------

import numpy as np
import matplotlib.pyplot as plt
import time

from bio_nica import bio_nica
from util import permutation_error, add_fill_lines

#---------------------------
# Load dataset
#---------------------------

print(f'Loading dataset...')

S = np.load('datasets/image/sources.npy')
X = np.load('datasets/image/mixtures.npy')

s_dim = S.shape[0]
x_dim = X.shape[0]
samples = S.shape[1]

#---------------------------
# Parameters
#---------------------------

epochs = 15; iters = epochs*samples

#---------------------------
# Bio-NICA
#---------------------------    

print('Testing Bio-NICA...')

bionica_err = np.zeros(iters)

# Shuffle data

S_permuted = np.zeros((s_dim,iters))
X_permuted = np.zeros((x_dim,iters))

for i_epoch in range(epochs):
    idx = np.random.permutation(samples)
    S_permuted[:,i_epoch*samples:(i_epoch+1)*samples] = S[:,idx[:]]
    X_permuted[:,i_epoch*samples:(i_epoch+1)*samples] = X[:,idx[:]]

bionica = bio_nica(s_dim, x_dim, 'image')
bionica_Y = np.zeros((s_dim,iters))

start_time = time.time()

for i_iter in range(iters):

    x = X_permuted[:,i_iter]
    bionica_Y[:,i_iter] = bionica.fit_next(x)

    if i_iter<=100:
        for j in range(s_dim):
            if sum(bionica_Y[0:i_iter,j])==0:
                bionica.flip_weights(j)

elapsed_time = time.time() - start_time

print(f'Elapsed time: {elapsed_time} seconds')

bionica_err = permutation_error(S_permuted,bionica_Y)

print(f'Final permutation error: {bionica_err[-1]}')

#---------------------------
# Performance plots
#---------------------------

print('Plotting...')

fig = plt.figure(figsize=(8,8))

plt.loglog(bionica_err, lw=3)
plt.grid()
plt.xlabel('Sample #')
plt.ylabel('Permutation error')
plt.xlim((10,iters))

plt.show()

#---------------------------
# Image transformations
#---------------------------

# Compute the inverse of the final permutation:

if i_epoch==epochs-1:
    inv_idx = np.argsort(idx)
    
Y = np.zeros((s_dim,samples))

for i_iter in range(samples):
    Y[:,i_iter] = bionica_Y[:,-samples+inv_idx[i_iter]]

figure = plt.figure(figsize=(8,6))

plt.subplot(3, 4, 1)
plt.imshow(S[0].reshape(252,252), cmap="gray")
plt.axis('off')
plt.title('Sources')

plt.subplot(3, 4, 5)
plt.imshow(S[1].reshape(252,252), cmap="gray")
plt.axis('off')

plt.subplot(3, 4, 9)
plt.imshow(S[2].reshape(252,252), cmap="gray")
plt.axis('off')

plt.subplot(3, 4, 2)
plt.imshow(X[0].reshape(252,252), cmap="gray")
plt.axis('off')
plt.title('Mixtures (a)')

plt.subplot(3, 4, 3)
plt.imshow(X[1].reshape(252,252), cmap="gray")
plt.axis('off')
plt.title('Mixtures (b)')

plt.subplot(3, 4, 6)
plt.imshow(X[2].reshape(252,252), cmap="gray")
plt.axis('off')

plt.subplot(3, 4, 7)
plt.imshow(X[3].reshape(252,252), cmap="gray")
plt.axis('off')

plt.subplot(3, 4, 10)
plt.imshow(X[4].reshape(252,252), cmap="gray")
plt.axis('off')

plt.subplot(3, 4, 11)
plt.imshow(X[5].reshape(252,252), cmap="gray")
plt.axis('off')

plt.subplot(3, 4, 4)
plt.imshow(Y[0].reshape(252,252), cmap="gray")
plt.axis('off')
plt.title('Recovered Sources')

plt.subplot(3, 4, 8)
plt.imshow(Y[1].reshape(252,252), cmap="gray")
plt.axis('off')

plt.subplot(3, 4, 12)
plt.imshow(Y[2].reshape(252,252), cmap="gray")
plt.axis('off')

figure.tight_layout(pad=-.05)

plt.show()