'''
Dependencies
'''
import numpy as np
from scipy.linalg import sqrtm
from tqdm import tqdm as pbar


def TensorMaximumEntropy(X):
    '''
    Given a tensor, returns the surrogate data by the TME method. 

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

        # Demean data along each axis
        X0 = X

        MT = np.mean(X0, axis=1)
        X0 = X0 - np.outer(MT, np.ones(N, ))

        MN = np.mean(X0, axis=0)
        X0 = X0 - np.outer(np.ones((T, )), MN)

        # Demeaned data X_bar and mean M
        X_bar = X0
        M = X - X_bar

        print('1/4')
        # Compute covariance matrices
        Sigma_T = np.array([[X_bar[i, :] @ X_bar[j, :].T for i in range(T)] for j in range(T)]) / N
        Sigma_N = np.array([[X_bar[:, i].T @ X_bar[:, j] for i in range(N)] for j in range(N)]) / T
        print('2/4')

        # Get eigenvalues and eigenvectors
        L_T, Q_T = np.linalg.eig(Sigma_T)
        L_N, Q_N = np.linalg.eig(Sigma_N)

        # Omit imaginary part
        L_T = np.real(L_T)
        L_N = np.real(L_N)

        # Sort eigenvalues and eigenvectors in descending order
        Tsortidx = np.argsort(-L_T) # Get indices of sorted eigenvalues (tensorSize = 2)
        Nsortidx = np.argsort(-L_N) 

        L_T, Q_T = L_T[Tsortidx], Q_T[Tsortidx]
        L_N, Q_N = L_N[Nsortidx], Q_N[Nsortidx]

        # Compute Kronecker sum and threshold
        d = np.array([L_T[i] + L_N for i in range(T)]).flatten()
        d[d < np.exp(-10)] = np.inf
        inv_d = 1 / d
        print('3/4')

        # Draw random samples from normal distribution
        vec_S0 = np.random.normal(size=(T * N))

        # Element-wise multiplication
        vec_S_star = np.multiply(np.sqrt(inv_d), vec_S0)    

        # Compute surrogates efficiently
        S_bar = np.real(np.array([np.kron(Q_T[t, :], Q_N) @ vec_S_star for t in range(T)]).flatten()).reshape((T, N))
        print('4/4')

        S = M + S_bar

        return S 






