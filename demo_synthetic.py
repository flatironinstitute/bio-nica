import numpy as np
import matplotlib.pyplot as plt
import time

from bio_nica import bio_NICA

print('Testing NICA algorithms on synthetic data...')

#----------
# Parameters
#----------
# Number of samples, epochs
samples = 10**5
n_epoch = 10
# Source and mixture dimensions
s_dim = 3
x_dim = 3
#----------

#---------------
# Generate data
#---------------

print(f'Generating {s_dim}-dimensional sparse uniform source data...')

# Generate sparse random samples

U = np.random.uniform(0,math.sqrt(48/5),(s_dim,samples)) # independent non-negative uniform source RVs with variance 1
B = np.random.binomial(1, .5, (s_dim,samples)) # binomial RVs to sparsify the source
S = U*B # sources

print(f'Generating {x_dim}-dimensional mixtures...')

# Generate random mixing matrix and mixtures

A = np.random.randn(x_dim,s_dim) # random mixing matrix
X = A@S
        
#--------------
# Run Bio-NICA
#--------------

print('Testing Bio-NICA...')

# Set learning rate

Y = np.zeros((s_dim,samples))
bio_nica = bio_nica(s_dim, x_dim, samples)

start_time = time.time()

for i in range(samples):
    x = X[:,i]
    Y[:,i] = bio_nica.fit_next(x)
    errs.append(subspace_error(sm.get_components(), U[:, :K]))
        
elapsed_time = time.time() - start_time

# Plotting...
print('Elapsed time: ' + str(elapsed_time))
# print('Final permutation error: ' + str(subspace_error(sm.get_components(), U[:, :K])))

print('Bio-NICA test complete!')

#-----------------
# Run 2-layer NSM
#-----------------

print('Testing 2-layer NSM...')

print('2-layer NSM test complete!')

#---------------------
# Run Nonnegative PCA
#---------------------

print('Testing Nonnegative PCA...')

print('Nonnegative PCA test complete!')
