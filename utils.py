'''
Date: 08.02.2021

Purpose: Helper functions for analyses
'''

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm as pbar

'''

Functions

'''

def read_cmp(file_path):
    """
    Read Blackrock Microsystems .cmp file into Python
    
    Parameters
    ----------
    file_path: str
        .cmp file path + name 
        
    Returns
    -------
    df_array: dataframe of shape (num electrodes, 5)
        [col (int), row (int), channel number (str), within_channel_num (int), global electrode number (int)]
    """
    # Open file, remove comments and remove other formatting we don't need
    with open(file_path) as f:
        temp = [line for line in f if not line.startswith('//')]     
    clean_lsts = [remove_cmp_formatting(l) for l in temp[1:]]

    df = pd.DataFrame(clean_lsts, columns=['array_col', 'array_row', 'channel_num', 'within_channel_num', 'global_enum']).dropna()
    
    # Convert columns to integers - errors='igore' return the column unchanged if it cannot be converted to a numeric type
    df_array = df.apply(pd.to_numeric, errors='ignore')
    
    return df_array


def localize_elecs(df, elecs, N=10, verbose=False):
    """
    Get the spatial location of electrodes on the array. 
    Set verbose=True to visualise the array.

    Parameters
    ----------
    df: pd.DataFrame
        .cmp file information
    elecs: lst 
        list of electrodes for which to get location on array
    N: int 
        number of rows and cols in array

    Returns
    -------
    elec_map: np.array
        each element in the electrode map is an electrode number 
    """
    
    elec_map = np.zeros((N, N))
    
    for e in elecs:
        if e in df['global_enum'].values: 
            # find row and column of electrode 
            i = int(df.loc[df['global_enum'] == e]['array_row'])
            j = int(df.loc[df['global_enum'] == e]['array_col'])
            elec_map[i, j] = e # put electrode number at this location
        else:
            if verbose:
                print(f'Electrode number {e} does not exist in array \n')
            continue
    
    if verbose: # display array with number of neurons
        fig, ax = plt.subplots(1, figsize=(6,6))
        
        ax.imshow(elec_map, cmap=cmap, interpolation='none', vmin=0, vmax=1, aspect='equal')
        # code to annotate the grid and draw white squares around each cell
        def rect(pos):
            r = plt.Rectangle(pos-0.5, 1,1, facecolor='none', edgecolor='w', linewidth=2)
            plt.gca().add_patch(r)
        x,y = np.meshgrid(np.arange(elec_map.shape[1]), np.arange(elec_map.shape[0]))
        m = np.c_[x[elec_map.astype(bool)], y[elec_map.astype(bool)]]
        for pos in m: 
            rect(pos)
        for i in range(len(elec_map)):
            for j in range(len(elec_map)):
                text = ax.text(j, i, int(elec_map[i, j]), ha='center', va='center', color='w')
    return elec_map


def remove_cmp_formatting(s):
    """
    Used in read_cmp() to remove formatting in .cmp file
    
    Parameters
    ----------
    s: str
        one line in the file
        
    Returns
    -------
    list of strings
    """
    for r in (('\t', ' '), ('\n', ''), ('elec', '')):
        s = s.replace(*r)       
    return s.split() 


def compute_stat_and_phys_distances(L, m1_unit_guide, pmd_unit_guide, m1_emap, pmd_emap, OTHER_ARRAY_D=50):
    ''' Compute pairwise correlations and corresponding physical distance. 
    
    Parameters
    ----------
    L: np array
        Latent variables (manifold) of dimension N x K, where K is the dimensionality of the manifold
    
    [name]_unit_guide: np array
        maps neurons to electrodes 
    
    [name]_emap: np array
        maps electrode number to location on array
        
    OTHER_ARRAY_D: int
        distance placeholder for when neurons are each on different arrays 
    
    Returns
    -------
    C: np array
        correlations between each unique neuron pair
    D: np array
        respective spatial distances between each neuron pair
    A: np array of str
        indicates to which array neuron pair belongs: 'M1', 'PMd' or '0A' if between comparison
    '''
    N = L.shape[0] 
    C = [] # Correlations
    D = [] # Spatial distances
    A = [] # On which array is the neuron pair
    Idx = [] # Which pairwise combination

    for i in range(N):
        for j in range(i+1, N): # NO repetition
            Idx.append([i,j])

            # Compute correlation or other distance metric between PCs 
            rho_ij, _ = stats.pearsonr(L[i, :], L[j, :])
            C.append(rho_ij)

            # Compute spatial distance between neurons
            if i < m1_unit_guide.shape[0] and j < m1_unit_guide.shape[0]: # If both neurons are located on M1 array (within)

                elec_i, elec_j = m1_unit_guide[i, 0], m1_unit_guide[j, 0] # Locate neuron on electrode
                loc_i, loc_j = np.array(np.where(m1_emap == elec_i)), np.array(np.where(m1_emap == elec_j))
                #print(f'Electrodes {elec_i:.0f} and {elec_j:.0f} on M1 at locations {(loc_i[0][0], loc_i[1][0])} and {(loc_j[0][0], loc_j[1][0])}.')
                D.append(np.linalg.norm(loc_i - loc_j))
                #print(f'Distance between electrodes {elec_i:.0f} and {elec_j:.0f} is {D[-1]:.2f}.')
                A.append('M1')

            elif i >= m1_unit_guide.shape[0] and j >= m1_unit_guide.shape[0]: # If both neurons are on the PMD arr (within)

                k = i - m1_unit_guide.shape[0]
                p = j - m1_unit_guide.shape[0]

                elec_i, elec_j = pmd_unit_guide[k, 0], pmd_unit_guide[p, 0] 
                loc_i, loc_j = np.array(np.where(pmd_emap == elec_i)), np.array(np.where(pmd_emap == elec_j))
                #print(f'Electrodes {elec_i:.0f} and {elec_j:.0f} on PMd at locations {(loc_i[0][0], loc_i[1][0])} and {(loc_j[0][0], loc_j[1][0])}.')
                D.append(np.linalg.norm(loc_i - loc_j))
                #print(f'Distance between electrodes {elec_i:.0f} and {elec_j:.0f} is {D[-1]:.2f}.')
                A.append('PMd')

            else: # If neuron i and j are located on different arrays
                D.append(OTHER_ARRAY_D)
                A.append('OA')

    return np.array(C), np.array(D), np.array(A), np.array(Idx)

