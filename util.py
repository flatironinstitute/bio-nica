# Title: util.py
# Description: Various utilities useful for online NICA tests
# Author: David Lipshutz (dlipshutz@flatironinstitute.org)

##############################
# Imports
import numpy as np

##############################

def permutation_error(S, Y):
    """
    Parameters:
    ====================
    S -- The data matrix of sources
    Y -- The data matrix of recovered sources
    Output:
    ====================
    err -- the (relative) Frobenius norm error
    """
    
    assert S.shape==Y.shape, "The shape of the sources S must equal the shape of the recovered sources Y"

    s_dim = S.shape[0]
    iters = S.shape[1]
    
    err = np.zeros(iters)
    
    perm = np.round(np.corrcoef(S,Y))[:s_dim,s_dim:]
                
    assert (perm@perm.T==np.eye(s_dim)).all(), "The recovered sources are not a clear permutation of the sources"
    
    for t in range(1,iters):

        error_t = (np.linalg.norm(S[:,t] - perm@Y[:,t])**2)/s_dim
        err[t] = err[t-1] + (error_t - err[t-1])/t
    
    return err


# def load_dataset(dataset_name, return_U=True, K=None):
#     '''
#     Parameters
#     ----------
#     dataset_name: str
#         name of dataset
#     return_U: bool
#         whether to also compute the eigenvetor matrix
#     Returns
#     -------
#         X: ndarray
#             generated samples
#         U: ndarray
#             ground truth eigenvectors
#         lam: ndarray
#             ground truth eigenvalues
#     '''

#     ld = loadmat(dataset_name)
#     fea = ld['fea']
#     X = fea.astype(np.float)
#     X -= X.mean(0)[None, :]

#     if return_U:
#         if K is None:
#             K = X.shape[-1]

#         pca = PCA(n_components=K, svd_solver='arpack')
#         pca.fit(X)
#         U = pca.components_.T
#         lam = pca.explained_variance_
#         X = X.T
#     else:
#         U = 0
#         lam = 0
#         X = X.T

#     return X, U, lam


# def generate_samples(K=None, N=None, D=None, method='spiked_covariance', options=None, scale_data=True,
#                      sample_with_replacement=False, shuffle=False, return_scaling=False):
#     '''
    
#     Parameters
#     ----------
#     D: int or None
#         number of features
    
#     K: int
#         number of components
    
#     N: int or 'auto'
#         number of samples, if 'auto' it will return all the samples from real data datasets
#     method: str
#         so far 'spiked_covariance' or 'real_data'
    
#     options: dict
#         specific of each method (see code)
#     scale_data: bool
#         scaling data so that average sample norm is one
#     shuffle: bool
#         whether to shuffle the data or not
#     return_scaling: bool
#         true if we want to get two additional output arguments, the centering and scaling
#     Returns
#     -------
#         X: ndarray
#             generated samples
#         U: ndarray
#             ground truth eigenvectors
#         sigma2: ndarray
#             ground truth eigenvalues
#         avg: ndarray
#             mean of X (only sometimes returned)
#         scale_factor: float
#             the amount by which the data was scaled (only sometimes returned)
#     '''
#     # Generate synthetic data samples  from a specified model or load real datasets
#     # here making sure that we use the right n when including n_test frames

#     if method == 'spiked_covariance':
#         if N == 'auto':
#             raise ValueError('N cannot be "auto" for spiked_covariance model')

#         if options is None:
#             options = {
#                 'lambda_K': 5e-1,
#                 'normalize': True,
#                 'rho': 1e-2 / 5,
#                 'return_U': True
#             }
#         return_U = options['return_U']

#         if N is None or D is None:
#             raise Exception('Spiked covariance requires parameters N and D')

#         rho = options['rho']
#         normalize = options['normalize']
#         if normalize:
#             lambda_K = options['lambda_K']
#             sigma = np.sqrt(np.linspace(1, lambda_K, K))
#         else:
#             slope = options['slope']
#             gap = options['gap']
#             sigma = np.sqrt(gap + slope * np.arange(K - 1, -1, -1))

#         U, _ = np.linalg.qr(np.random.normal(0, 1, (D, K)))

#         w = np.random.normal(0, 1, (K, N))
#         X = np.sqrt(rho) * np.random.normal(0, 1, (D, N))

#         X += U.dot((w.T * sigma).T)
#         sigma2 = (sigma ** 2)[:, np.newaxis]

#     elif method == 'real_data':
#         if options is None:
#             options = {
#                 'filename': './datasets/MNIST.mat',
#                 'return_U': True
#             }
#         return_U = options['return_U']
#         filename = options['filename']

#         X, U, sigma2 = load_dataset(filename, return_U=return_U, K=K)

#         if N != 'auto':
#             if N > X.shape[-1]:
#                 if sample_with_replacement:
#                     print('** Warning: You are sampling real data with replacement')
#                 else:
#                     raise Exception("You are sampling real data with replacement "
#                                     "but sample_with_replacement flag is set to False")

#             X = X[:, np.arange(N) % X.shape[-1]]

#     else:
#         assert 0, 'Specified method for data generation is not yet implemented!'

#     # center data
#     avg = X.mean(1)[:, None]
#     X -= avg
#     if scale_data:
#         scale_factor = get_scale_data_factor(X)
#         X, U, sigma2 = X * scale_factor, U, sigma2 * (scale_factor ** 2)
#     else:
#         scale_factor = 1

#     if shuffle:
#         print('Shuffling data!')
#         X = X[:,np.random.permutation(X.shape[-1])]

#     if return_scaling:
#         if return_U:
#             return X, U, sigma2, avg, scale_factor
#         else:
#             return X, avg, scale_factor
#     else:
#         if return_U:
#             return X, U, sigma2
#         else:
#             return X