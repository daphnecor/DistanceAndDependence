"""

Dependencies

"""

import numpy as np
import pandas as pd
import seaborn as sns
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import matplotlib.patches as mpatches
cs = ['#732514', '#FEB312', '#233A6A', '#545340', '#4E81AF', '#183D51']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=cs)


"""

Functions

"""

def compare_r_pcs(pcs, r, m1_emap, m1_unitguide, pmd_emap, pmd_unitguide, norm=None):
    '''
    Compute the distance of the first r principal components for each pair of neurons.
    
    Parameters
    ----------
    pcs: np.array
        principal components
    r: int
        the number of principal components (cols) to compare
    norm: int or str
        by default takes the euclidean norm (norm=None), if norm=1 the L1 norm (abs) will be used
    '''
    # Compare neurons within M1 
    n_m1 = m1_unitguide.shape[0] # Number of neurons on M1 array
    M_m1 = pcs[0:n_m1, :r] # Take the first r columns of the principal components matrix (on M1 array)
    # Don't care where the neuron comes from (2d-array), 
    # we only want to know how spaial distance and weight distance relate (1d-array)
    m1_within_S_dist, m1_within_W_dist = [], []
    
    for i in range(n_m1): 
        
        elec_i = m1_unitguide[i, 0] # Neuron i is on this electrode
        loc_i  = np.array(np.where(m1_emap == elec_i)) # location of electrode on array
        
        # Compare this neuron's weights (row) to all other neurons within array (j != i)
        for j in range(i+1, n_m1):
            
            elec_j = m1_unitguide[j, 0] # Neuron j is on this electrode
            loc_j  = np.array(np.where(m1_emap == elec_j))

            # Compute spatial distance between two neurons on array
            m1_within_S_dist.append(LA.norm(loc_i - loc_j))

            # Compute PC weight distance between the neurons on array
            m1_within_W_dist.append(LA.norm(M_m1[i, :] - M_m1[j, :], ord=norm))
        
    # Compare neurons within PMd
    n_pmd = pmd_unitguide.shape[0] # Number of neurons on PMd array
    M_pmd = pcs[n_m1:, :r] 
    pmd_within_S_dist, pmd_within_W_dist = [], []    
    
    for i in range(n_pmd):
        
        elec_i = pmd_unitguide[i, 0] # Neuron i is on this electrode
        loc_i  = np.array(np.where(pmd_emap == elec_i)) # location of electrode on array

        # Compare this neuron's weights (row) to all other neurons within array (j != i)
        for j in range(i+1, n_pmd):

            elec_j = pmd_unitguide[j, 0] # Neuron j is on this electrode
            loc_j  = np.array(np.where(pmd_emap == elec_j))

            # Compute spatial distance between two neurons on array
            pmd_within_S_dist.append(LA.norm(loc_i - loc_j))

            # Compute PC weight distance between the neurons on array
            pmd_within_W_dist.append(LA.norm(M_pmd[i, :] - M_pmd[j, :], ord=norm))
            
    # Compare neurons between M1 and PMd
    between_W_dist = []
    
    for i in range(n_pmd):
        for j in range(n_m1):
            between_W_dist.append(LA.norm(M_pmd[i, :] - M_m1[j, :], ord=norm))
            
    return np.array(m1_within_S_dist), np.array(m1_within_W_dist), np.array(pmd_within_S_dist), np.array(pmd_within_W_dist), np.array(between_W_dist)


def standardized_hist(W_dist_se, W_dist_sa, W_dist_oa, rand_W_dist_se, rand_W_dist_sa, rand_W_dist_oa, i, dividesqrt=False, plot_separate=False):
    ''' Plots the different groups through dividing all entries of a group by the max value in that group.
    
    Parameters
    ----------
    TODO!
    '''

    binz = np.arange(0, 1.51, 0.01)
    xbar = [f'{b:.2f}' for b in binz]
    intvals = np.append(binz, np.inf)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
    #fig.suptitle(f'Weight distances with the highest {r} PCs', fontsize=15)
    
    if dividesqrt:
        W_dist_se = W_dist_se / np.sqrt(i)
        W_dist_sa = W_dist_sa / np.sqrt(i)
        W_dist_oa = W_dist_oa / np.sqrt(i)
    
    # KS tests
    p_se_sa, _ = ks_2samp(W_dist_se, W_dist_sa, alternative='less', mode='auto')
    p_se_oa, _ = ks_2samp(W_dist_se, W_dist_oa, alternative='less', mode='auto')
    p_sa_oa, _ = ks_2samp(W_dist_sa, W_dist_oa, alternative='less', mode='auto')

    # Compute all bins
    oa_bins = [np.sum((W_dist_oa >= intvals[i]) & (W_dist_oa < intvals[i + 1])) for i in range(len(binz))]
    oa_bins = oa_bins / max(oa_bins) * -1 # flip to the other side for visualization purposes

    sa_bins = [np.sum((W_dist_sa >= intvals[i]) & (W_dist_sa < intvals[i + 1])) for i in range(len(binz))]
    sa_bins = sa_bins / max(sa_bins)

    se_bins = [np.sum((W_dist_se >= intvals[i]) & (W_dist_se < intvals[i + 1])) for i in range(len(binz))]
    se_bins = se_bins / max(se_bins) 

    # Plot 
    axs[0].bar(x=xbar, height=oa_bins, label='Other array', alpha=0.8, color=cs[2])
    axs[0].bar(x=xbar, height=sa_bins, label='Same array',   alpha=0.8, color=cs[1])
    axs[0].bar(x=xbar, height=se_bins, label='Same electrode', alpha=0.8, color=cs[0])
    
    axs[0].set_xlabel(r'$w_{dist}$')
    axs[0].set_xticks([f'{i/10:.2f}' for i in range(16)])
    if dividesqrt: ax.set_xticks([]) # Then the x axis doesn't make sense anymore
    axs[0].annotate(f'Same elec to same array | p={p_se_sa:.3f}', xy=(0.65, 0.9), xycoords=axs[0].transAxes)
    axs[0].annotate(f'Same elec to other array  | p={p_se_oa:.3f}', xy=(0.65, 0.8), xycoords=axs[0].transAxes)
    axs[0].annotate(f'Same array to other array | p={p_sa_oa:.3f}', xy=(0.65, 0.7), xycoords=axs[0].transAxes)


    # Now the random ones
    # KS tests
    p_se_sa, _ = ks_2samp(rand_W_dist_se, rand_W_dist_sa, alternative='less', mode='auto')
    p_se_oa, _ = ks_2samp(rand_W_dist_se, rand_W_dist_oa, alternative='less', mode='auto')
    p_sa_oa, _ = ks_2samp(rand_W_dist_sa, rand_W_dist_oa, alternative='less', mode='auto')

    # Compute all bins
    rand_oa_bins = [np.sum((rand_W_dist_oa >= intvals[i]) & (rand_W_dist_oa < intvals[i + 1])) for i in range(len(binz))]
    rand_oa_bins = rand_oa_bins / max(rand_oa_bins) * -1 # flip to the other side for visualization purposes

    rand_sa_bins = [np.sum((rand_W_dist_sa >= intvals[i]) & (rand_W_dist_sa < intvals[i + 1])) for i in range(len(binz))]
    rand_sa_bins = rand_sa_bins / max(rand_sa_bins)

    rand_se_bins = [np.sum((rand_W_dist_se >= intvals[i]) & (rand_W_dist_se < intvals[i + 1])) for i in range(len(binz))]
    rand_se_bins = rand_se_bins / max(rand_se_bins) 


    axs[1].bar(x=xbar, height=rand_oa_bins, label='Other array', alpha=0.9, color=cs[2])
    axs[1].bar(x=xbar, height=rand_sa_bins, label='Same array',   alpha=0.9, color=cs[1])
    axs[1].bar(x=xbar, height=rand_se_bins, label='Same electrode', alpha=0.9, color=cs[0])
    axs[1].set_xlabel(r'Random $w_{dist}$')
    axs[1].set_xticks([f'{i/10:.2f}' for i in range(11)])
    axs[1].legend(loc='center right', bbox_to_anchor=(1, 0.2))

    sns.despine()
    plt.show();

    if plot_separate:
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 6))

        axs[0].bar(x=xbar, height=oa_bins*-1, label='Other array', color=cs[2])
        axs[0].set_xticks([])

        axs[1].bar(x=xbar, height=sa_bins, label='Same array', alpha=1, color=cs[1])
        axs[1].set_xticks([])

        axs[2].bar(x=xbar, height=se_bins,  label='Same electrode', alpha=1, color=cs[0])
        axs[2].set_xticks([f'{i/10:.2f}' for i in range(10)])

        sns.despine()
        plt.show();
        

def compare_one_to_all(pcs, emap, unitguide, n_arr, i):
    '''
    Compares PCs from one neuron (row i) to PCs to all other neurons (rows j!=i) within array with range of r PCs (cols).
    
    pcs: np.array (N x N)
        PCA done on all neurons
    
    n_arr: int
        number of neurons on array
    i: int
        neuron with which to compare all others
    '''
    # Define color scheme to plot 
    cs = ['#438654', '#518e60', '#60976d', '#6ea079', '#7da886', '#8bb192', '#9ab99e', '#a8c1ab', '#b6cab7', '#c5d3c4', '#d3dbd0', '#f0ece9']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=cs)
    
    m1_within_S_dist = np.zeros(n_arr) 
    m1_within_W_dist = np.zeros((pcs.shape[0], n_arr))
    
    elec_ni = unitguide[i, 0]
    loc_ni  = np.asarray(np.where(emap == elec_ni))
    
    for r in np.arange(0, pcs.shape[0], 1):
    
        M = pcs[:, :r] # Take first r PCs

        # Compare this neuron's weights (row) to all other neurons within array (j != i)
        for j in range(i, n_arr):

            elec_nj = unitguide[j, 0]
            loc_nj  = np.asarray(np.where(emap == elec_nj))

            # Get distance between two neurons on array
            m1_within_S_dist[j] = LA.norm(loc_ni - loc_nj)

            # Compute PC weight distance between the neurons
            m1_within_W_dist[r, j] = LA.norm(M[i, :] - M[j, :])   
    
    # Group by spatial distance
    mask_1 = np.where(m1_within_S_dist <= 1)[0]
    mask_2 = np.where(m1_within_S_dist <= 2)[0]
    mask_3 = np.where(m1_within_S_dist <= 3)[0]
    mask_4 = np.where(m1_within_S_dist <= 4)[0]
    mask_5 = np.where(m1_within_S_dist <= 5)[0]
    mask_6 = np.where(m1_within_S_dist <= 6)[0]
    mask_7 = np.where(m1_within_S_dist <= 7)[0]
    mask_8 = np.where(m1_within_S_dist <= 8)[0]
    mask_9 = np.where(m1_within_S_dist <= 9)[0]
    mask_10 = np.where(m1_within_S_dist <= 12)[0]
    
    fig, ax = plt.subplots(1, figsize=(12, 6))
    
    ax.plot(m1_within_W_dist[:, mask_10], )
    ax.plot(m1_within_W_dist[:, mask_9], )
    ax.plot(m1_within_W_dist[:, mask_8], )
    ax.plot(m1_within_W_dist[:, mask_7], )
    ax.plot(m1_within_W_dist[:, mask_6], )
    ax.plot(m1_within_W_dist[:, mask_5], )
    ax.plot(m1_within_W_dist[:, mask_4], )
    ax.plot(m1_within_W_dist[:, mask_3], )
    ax.plot(m1_within_W_dist[:, mask_2], )
    ax.plot(m1_within_W_dist[:, mask_1], )
    
    ax.set_xlabel(f'Number of PCs used to compute the distance')
    ax.set_ylabel(f'Weight distance to neuron {i+1}')

    # Make legend
    farsquare = mpatches.Patch(color=cs[9], label=r'Small spatial distance')
    closesquare = mpatches.Patch(color=cs[0], label=r'Large spatial distance')
    ax.legend(handles=[farsquare, closesquare], bbox_to_anchor=(1., 0.8))
    sns.despine()
    plt.show();


def make_groups(S_dist_m1, W_dist_m1, S_dist_pmd, W_dist_pmd):
    '''Group weight distances of the neurons based on spatial distance in the brain.'''

    # First we merge the spatial distances obtained from the M1 and PMd arrays
    S_dist_both = np.concatenate((S_dist_m1, S_dist_pmd))

    # Then we merge the weight distances obtained from both arrays
    W_dist_both = np.concatenate((W_dist_m1, W_dist_pmd))

    # Now we can make groups based on distance
    same_elec_idx  = np.where(S_dist_both == 0) # Zero distance
    same_array_idx = np.where(S_dist_both != 0) # On the same array

    # Use the indices to group the weight distances
    W_dist_se = W_dist_both[same_elec_idx]
    W_dist_sa = W_dist_both[same_array_idx]
    
    return S_dist_both, W_dist_se, W_dist_sa
