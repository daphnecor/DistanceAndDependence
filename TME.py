'''
Dependencies
'''
import numpy as np
from scipy.linalg import sqrtm


def TensorMaximumEntropy(X):
    '''
    Given a 3D tensor, returns the surrogate data by the TME method. 

    Parameters
    ----------
    X: Firing rates in the form T x N x C (times, neurons, conditions/trials) or T X (N * C)

    Returns
    -------
    S_bar: The surrogate data with same primary features (i.e. mean and covariance) as the true dataset. 
        

    For details, see ONLINE METHODS section in (Elsayed & Cunningham, 2017) - doi:10.1038/nn.4617 
        and matlab implementation - https://github.com/gamaleldin/TME
    
    '''

    dim = len(X.shape)

    if dim < 2:
        raise ValueError('TME requires data with dimensionality > 2.')

    if dim == 2: # If input data is trial-averaged or concatenated
        T, N = X.shape

        # (1) Quantify primary features across different modes of the data
        # Mean-center data sequentially: compute and subtract mean across each mode
        X0 = X

        MT = np.mean(X0, axis=1)
        X0 = X0 - MT.reshape((T, 1)) @ np.ones((1, N))

        MN = np.mean(X0, axis=0)
        X0 = X0 - (MN.reshape((N, 1)) @ np.ones((1, T))).T

        X_bar = X0
        M = X - X_bar

        # Compute covariance matrices
        # Compute dot product between each col sequentially 
        print('Computing covariance marices [this may take a while]')
        Sigma_T = compute_cov_naive(X_bar) 
        Sigma_N = X_bar.T @ X_bar
        
        # Get eigenvalues and eigenvectors
        L_T, Q_T = np.linalg.eig(Sigma_T)
        L_T = np.diag(L_T)

        L_N, Q_N = np.linalg.eig(Sigma_N)
        L_N = np.diag(L_N)
        
        # Compute the maximum entropy distribution: \matcal{N}(vec(M), \Psi)
        Psi = 1/2 * np.kron(Q_T, Q_N) @ (np.linalg.inv(kron_add(L_T, L_N))) @ np.kron(Q_T, Q_N).T

        # Sample noise vec(S0) from \mathcal{N}(O, I)
        O = np.zeros((T*N)) # Mean
        I = np.identity((T*N)) # Cov
        S0 = np.random.multivariate_normal(mean=O, cov=I) # Sample

        # Vectorize         
        vec_S0 = S0.flatten()
        vec_M = M.flatten()

        # Compute S
        vec_S = vec_M + sqrtm(Psi) @ S0

        # Reshape shuffled data to original form
        S_bar = vec_S.reshape((T, N))

        
    elif dim == 3:
        T, N, C = X.shape

        # (1) Quantify primary features across different modes of the data
        # Mean-center data sequentially: compute and subtract mean across each mode
        X0 = X

        MT = np.mean(X0, axis=(1, 2))
        X0 = X0 - (MT.reshape((T, 1)) @ np.ones(shape=(1, N))).reshape((T, N, 1)) @ np.ones((1, C))

        MN = np.mean(X0, axis=(0, 2))
        X0 = X0 - (np.ones(shape=(T, 1)) @ MN.reshape((1, N))).reshape((T, N, 1)) @ np.ones((1, C))

        MC = np.mean(X0, axis=(0, 1))
        X0 = X0 - (np.ones(shape=(T, 1)) @ np.ones(shape=(1, N))).reshape((T, N, 1)) @ MC.reshape((1, C))

        X_bar = X0 # The demeaned data
        M = X - X_bar # The mean

        Sigma_T = np.tensordot(X_bar, X_bar, axes=([1, 2],[1, 2]))
        Sigma_N = np.tensordot(X_bar, X_bar, axes=([0, 2],[0, 2]))
        Sigma_C = np.tensordot(X_bar, X_bar, axes=([0, 1],[0, 1]))

        # (2) Generate surrogate data with the TME method
        # Get the eigenvalues L and eigenvectors Q
        L_T, Q_T = np.linalg.eig(Sigma_T)
        L_T = np.diag(L_T)

        L_N, Q_N = np.linalg.eig(Sigma_N)
        L_N = np.diag(L_N)

        L_C, Q_C = np.linalg.eig(Sigma_C)
        L_C = np.diag(L_C)

        # Compute the maximum entropy distribution: \matcal{N}(vec(M), \Psi)
        Psi = 1/2 * (np.kron(np.kron(Q_C, Q_N), Q_T)) @ np.linalg.inv((kron_add(kron_add(L_C, L_N), L_T))) \
        @ (np.kron(np.kron(Q_C, Q_N), Q_T)).T

        # Sample noise vec(S0) from \mathcal{N}(O, I)
        O = np.zeros((T*N*C)) # Mean
        I = np.identity((T*N*C)) # Cov
        S0 = np.random.multivariate_normal(mean=O, cov=I) # Sample

        # Vectorize         
        vec_S0 = S0.flatten()
        vec_M = M.flatten()

        # Compute S
        vec_S = vec_M + sqrtm(Psi) @ S0

        # Reshape shuffled data to 3D tensor in original form
        S_bar = vec_S.reshape((T, N, C))

    
    return S_bar


def kron_add(A, B):
    ''' Performs kronecker addition '''
    return np.kron(A, np.ones(np.asarray(B).shape)) + np.kron(np.ones(np.asarray(A).shape), B)

def compute_cov_naive(X):
    ''' Computes covariance of fat matrices using for loops (memory can't handle the size)
        Caution: may take a long time'''
    T, N = X.shape
    Cov = np.zeros((T, T))
    for i in pbar(range(T)):
        for j in range(T):
            Cov[i, j] = X[i, :] @ X[j, :]        
    return Cov







