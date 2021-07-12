"""

Dependencies

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from sklearn import decomposition
cmap = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import ks_2samp

from ipywidgets import interact

sns.set_style('white')
sns.set_context('notebook', font_scale=1.35)

cmap = sns.light_palette(color='#314ac0', as_cmap=True)
div_cmap = sns.diverging_palette(250, 150, as_cmap=True)
dist_cmap = sns.light_palette('#092d68', 9)

cs = ['#314ac0', '#ef3737', '#5ae09e', '#EBE12E', '#3D7E43', '#2D2F39', '#4D7399', '#CDBCA5']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=cs)

"""

Functions

"""

def make_raster(spike_trains):
	'''
	Plots a raster given a set of (binary) spike trains.
	
	Parameters
	----------
	spike_trains: np.array 
		set of binary arrays in the form [# timepoints, # neurons]
	'''
	if np.size(spike_trains) == 0:
		print('This array is empty')
		return
	
	# In the case some entries > 1 convert them to 1
	binary_trains = np.isin(spike_trains, range(1, 10)).astype(np.uint8) 
	
	# Scale length of plot by number of neurons   
	if spike_trains.shape[1] < 2: 
		flen = 1
	else: 
		flen = spike_trains.shape[1]/5
	
	fig, ax = plt.subplots(1, figsize=(12, flen), dpi=80)
	for i in range(spike_trains.shape[1]): 
		y_val = i + 1 
		spike_train_i = binary_trains[:, i] * y_val
		spike_train_i = [float('nan') if x==0 else x for x in spike_train_i] 

		plt.scatter(range(spike_trains.shape[0]), spike_train_i,  marker='|', c='k', s=50);
		ax.set_title(f'Raster plot of {spike_trains.shape[1]} neuron(s)', fontsize=15)
		ax.set_xlabel('time', fontsize=13)
		ax.set_ylabel('neuron', fontsize=13)


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


def elecs_to_neurons(elec_map, unit_guide, elecs=list(range(1, 97)), N=10):
	"""
	Get mapping between electrode number amount of neurons per electrode
	
	Parameters  
	----------
	elec_map: np.array 
		NxN array with electrode numbers as elements

	unit_guide: np.array 
		specifies the number of neurons each electrode covers 
		note that this mapping is the same for all trials

	Returns
	-------
	cell_distrib: np.array
		distribution of neurons on given array

	cells_on_arr: np.array
		array with number of neurons for each electrode
	"""

	cells_on_arr = np.zeros((N, N))
	cell_distrib = []

	for e in elecs: 
		e_indices = np.where(unit_guide[:, 0] == e)

		if np.size(e_indices) == 0: 
			cell_distrib.append(0)
		else:
			neurons_at_e = max(unit_guide[:, 1][e_indices])
			
			cell_distrib.append(neurons_at_e)
			e_loc = np.where(elec_map == e)
			cells_on_arr[e_loc] = neurons_at_e

	return cell_distrib, cells_on_arr


def compare_pc_weights(m1_arr, m1_ug, pmd_arr, pmd_ug, w):
	"""
	Compares the PC weights across distance
	
	Parameters
	----------
	m1_arr: np.array
	   electrode map containing the spatial location of the m1 electrodes 
	
	m1_ug: 
		unit guide of the m1 array
	
	pmd_arr: np.array
		electrode map containing the spatial location of the m1 electrodes

	pmd_ug: 
		unit guide of the pmd array
	
	w: np.array 
		vector with the weights or loadings of the first principal component (all neurons)
	
	Returns
	-------
	df: pandas dataframe 
		dataframe that can be used to plot distributions directly with seaborn by group

	Note that this assumes that the weight vector is constructed such that all M1 weights are first, 
	and all PMd weights are second like [M1, M1, ... , PMd , PMd]
	"""
    
	# compare within m1 array
	within_m1_dist, within_m1_w = [], [] 

	for i in range(len(m1_ug[:, 0])): # loop along neurons 
		# find electrode that corresponds to this neuron
		elec1 = m1_ug[i, 0]
		loc1 = np.where(m1_arr == elec1) # find neuron location on array 

		for j in range(i+1, len(m1_ug[:, 0])): # compare to all other electrodes (j!=i)        
			# find electrode location of this neuron within same array
			elec2 = m1_ug[j, 0]
			loc2 = np.where(m1_arr == elec2)

			# find euclidean distance between two neurons on array
			dst = distance.euclidean(loc1, loc2)

			within_m1_dist.append(dst) 
			within_m1_w.append(np.abs(w[j] - w[i])) 

	# compare within pmd array
	within_pmd_dist, within_pmd_w = [], [] 

	for i in range(len(pmd_ug[:, 0])): # loop along neurons 
		# find electrode that corresponds to this neuron
		elec1 = pmd_ug[i, 0]
		loc1 = np.where(pmd_arr == elec1) # find neuron location on array 

		for j in range(i+1, len(pmd_ug[:, 0])): # compare to all other electrodes (j!=i)        
			# find electrode location of this neuron within same array
			elec2 = pmd_ug[j, 0]
			loc2 = np.where(pmd_arr == elec2)

			# find euclidean distance between two neurons on array
			dst = distance.euclidean(loc1, loc2)

			within_pmd_dist.append(dst) 
			within_pmd_w.append(np.abs(w[j] - w[i])) 
	
	# compare pmd to m1
	pmd_m1_w = []
	
	for i in range(len(pmd_ug[:, 0])): # loop along neurons 
		# compare to all neurons in other array
		for j in range(len(m1_ug[:, 0])):
			# compare all weights from main to all weights from other
			pmd_m1_w.append(np.abs(w[pmd_ug.shape[0]+j] - w[i]))

	df = pd.DataFrame({'distance':np.concatenate((within_m1_dist, within_pmd_dist, np.full(len(pmd_m1_w), np.nan)), axis=0), 
		'w_diff': np.concatenate((within_m1_w, within_pmd_w, pmd_m1_w), axis=0)})

	# group them to make plotting easier
	df['array'] = np.nan
	df['array'].iloc[:len(within_m1_w)] = 'M1'
	df['array'].iloc[len(within_m1_w): len(within_m1_w) + len(within_pmd_w)] = 'PMd'
	df['group'] = df['distance'].apply(lambda d: 'same elec' if d == 0 else ('same array' if d > 0 else ('other array')))
	df['within arr distance'] = pd.cut(df['distance'], bins=[0, 0.001, 3, 6, 9, 16], labels=['0', '0 - 3','3 - 6','6 - 9', '> 9'])
	return df
	

def display_grid(arr):
	''' Displays the number of neurons per electrode on array.'''
	
	fig, ax = plt.subplots(1, figsize=(5,5), dpi=80)
	ax.imshow(arr, cmap='viridis')
	for i in range(len(arr)):
		for j in range(len(arr)):
			text = ax.text(j, i, int(arr[i, j]), ha='center', va='center', color='w')


def compare_self_global_mani(m1_pcs, pmd_pcs, m1pmd_pcs, end_M1, pc_dim, verbose=True):
    ''' 
    Compare self to global manifold 
    Caution! The PCs must be on the COLUMNS and neurons on the ROWS
    
    Parameters
    ----------
    m1_pcs: np.array
        PCA done on m1 neural firing rates alone
    pmd_pcs: np.array
        PCA done on pmd neural firing rates alone
    both_pcs:
        PCA done on all neural firing rates
    pc_dim: int
        Principal Component vector to use
    end_M1: int
    	number of neurons that belong to M1 arr
        
    Return
    ------
    Scatterplot & regplot with correlation coefficient r
    '''

    self_pmd = np.array([pmd_pcs[:, pc_dim]])
    glob_pmd = np.array([m1pmd_pcs[end_M1:, pc_dim]])
    self_glob_pmd = np.vstack((self_pmd, glob_pmd)).T

    self_m1 = np.array([m1_pcs[:, pc_dim]])
    glob_m1 = np.array([m1pmd_pcs[0:end_M1, pc_dim]])
    self_glob_m1 = np.vstack((self_m1, glob_m1)).T
    
    self_glob_mani = np.vstack((self_glob_pmd, self_glob_m1))
    r = np.corrcoef(self_glob_mani[:, 0], self_glob_mani[:, 1])[0,1]
    
    if verbose:
	    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
	    fig.suptitle(f'Comparison self vs global manifold for PC {pc_dim+1}', fontsize=15, y=1.03)

	    ax1.set_title(f'r = {round(r,4)}')
	    ax1.plot(self_glob_pmd[:, 0], self_glob_pmd[:, 1], 'o', label='PMd')
	    ax1.plot(self_glob_m1[:, 0], self_glob_m1[:, 1], 'o', label='M1')
	    ax1.plot([-0.5, 0.5], [-0.5, 0.5], '--k', label=r'$r=1$')
	    ax1.set_xlabel('Weight on self-manifold')
	    ax1.set_ylabel('Weight on global-manifold')
	    ax1.set_xticks(np.arange(-0.5, 0.6, 0.1))
	    ax1.set_yticks(np.arange(-0.5, 0.6, 0.1))
	    ax1.legend()

	    sns.regplot(self_glob_mani[:, 0], self_glob_mani[:, 1], ax=ax2, label=r'true $r$')
	    ax2.set_xticks(np.arange(-0.5, 0.6, 0.1))
	    ax2.set_yticks(np.arange(-0.5, 0.6, 0.1))
	    ax2.set_xlabel('Weight on self-manifold')
	    ax2.legend()
	    sns.despine()

    return r


def sort_pcs(PCs, end_M1, dim):
    '''
    Sort the principal components loadings by the indices of the dimension 'by' in descending order.
    Caution! The PCs must be on the COLUMNS in order for this to work.
    
    Parameters
    ----------
    PCs: np.array
        n_neurons (features) x n_components
    
    end_M1: int
		first x neurons belong to M1

    dim: number of the pc by which you want to sort the others
    
    Returns
    -------
    Ordered PCs with an empty row between M1 and PMd neurons.
    '''
  
    m1_pcs = PCs[0:end_M1, :]
    pmd_pcs = PCs[end_M1:, :]
    
    # take the k th pc (the one by which you want to sort)
    m1_k_pc = abs(m1_pcs[:, dim])
    pmd_k_pc = abs(pmd_pcs[:, dim])
    
    # get indices that get this pc in descending order
    desc_order_idx_m1 = np.argsort(m1_k_pc)[::-1]
    desc_order_idx_pmd = np.argsort(pmd_k_pc)[::-1]
    
    # sort all pcs in descending order
    m1_pcs_ordered = m1_pcs[desc_order_idx_m1]
    pmd_pcs_ordered = pmd_pcs[desc_order_idx_pmd]

    return np.concatenate((m1_pcs_ordered, np.full((2, m1_pcs_ordered.shape[1]), np.nan), pmd_pcs_ordered), axis=0)


def generate_weight_distrib(m1_elecmap, pmd_elecmap, td, PCs, pc_dim, detailed_plots=False):
    
    # compare weights of pc number K
    df = compare_pc_weights(m1_arr=m1_elecmap, m1_ug=td['M1_unit_guide'][0], pmd_arr=pmd_elecmap, pmd_ug=td['PMd_unit_guide'][0], w=PCs[:, pc_dim])
    
    df_m1  = df.loc[df['array'].isin(['M1', np.nan])]
    df_pmd = df.loc[df['array'].isin(['PMd', np.nan])]
    df_same_arr = df.loc[df['group'] == 'same array']
    # make within categories
    m1_cats = df_same_arr.loc[df['array']=='M1']
    pmd_cats = df_same_arr.loc[df['array']=='PMd']

    # global settings
    binz = np.arange(0, 0.5, 0.005)
    xbar = [str(bin) for bin in binz]
    intvals = np.append(binz, np.inf)

    if detailed_plots:
    
	    # Generate 2 x 2 plot
	    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
	    fig.suptitle(f'pc number {pc_dim + 1}', fontsize=18)
	    
	    # M1
	    se = df_m1.loc[df_m1['group'] == 'same elec']['w_diff'].values
	    sa = df_m1.loc[df_m1['group'] == 'same array']['w_diff'].values
	    oa = df_m1.loc[df_m1['group'] == 'other array']['w_diff'].values

	    ax1.set_title('pc weight distribution M1, sorted by group')
	    oa_bins = [np.sum((oa >= intvals[i]) & (oa < intvals[i + 1])) for i in range(len(binz))]
	    oa_bins = oa_bins / max(oa_bins)
	    ax1.bar(x=xbar, height=oa_bins, label='Other array')

	    sa_bins = [np.sum((sa >= intvals[i]) & (sa < intvals[i + 1])) for i in range(len(binz))]
	    sa_bins = sa_bins / max(sa_bins)
	    ax1.bar(x=xbar, height=sa_bins, label='Same array', alpha=.7)

	    se_bins = [np.sum((se >= intvals[i]) & (se < intvals[i + 1])) for i in range(len(binz))]
	    se_bins = se_bins / max(se_bins)
	    ax1.bar(x=xbar, height=se_bins,  label='Same electrode', alpha=.7)
	    ax1.set_xlabel(r'$w_{diff}$')
	    ax1.set_xticks([str(i / 10) for i in range(0, 6)])
	    ax1.legend()
	    
	    # PMD
	    se = df_pmd.loc[df_pmd['group'] == 'same elec']['w_diff'].values
	    sa = df_pmd.loc[df_pmd['group'] == 'same array']['w_diff'].values
	    oa = df_pmd.loc[df_pmd['group'] == 'other array']['w_diff'].values
	    
	    ax2.set_title('pc weight distribution PMd, sorted by group')
	    oa_bins = [np.sum((oa >= intvals[i]) & (oa < intvals[i + 1])) for i in range(len(binz))]
	    oa_bins = oa_bins / max(oa_bins)
	    ax2.bar(x=xbar, height=oa_bins, label='Other array')

	    sa_bins = [np.sum((sa >= intvals[i]) & (sa < intvals[i + 1])) for i in range(len(binz))]
	    sa_bins = sa_bins / max(sa_bins)
	    ax2.bar(x=xbar, height=sa_bins, label='Same array', alpha=.7)

	    se_bins = [np.sum((se >= intvals[i]) & (se < intvals[i + 1])) for i in range(len(binz))]
	    se_bins = se_bins / max(se_bins)
	    ax2.bar(x=xbar, height=se_bins,  label='Same electrode', alpha=.7)
	    ax2.set_xlabel(r'$w_{diff}$')
	    ax2.set_xticks([str(i / 10) for i in range(0, 6)])
	    ax2.legend()
	    sns.despine()
	    
	    # within array plots with seaborn (not scaled)
	    sns.histplot(m1_cats, x='w_diff', hue='within arr distance', bins=binz, stat='count', ax=ax3, multiple='stack')
	    ax3.set_title('Within M1 array weight differences')
	    ax3.set_xlabel(r'$w_{diff}$')

	    sns.histplot(pmd_cats, x='w_diff', hue='within arr distance', bins=binz, stat='count', ax=ax4, multiple='stack')
	    ax4.set_title('Within PMd array weight differences')
	    ax4.set_xlabel(r'$w_{diff}$')
	    sns.despine()
	    plt.tight_layout()
	    plt.show()

    # ALL
    fig, ax = plt.subplots(1, figsize=(11, 6))
    ax.set_title(f'PC {pc_dim+1} weight differences | all neurons, sorted by group')
    se = df.loc[df['group'] == 'same elec']['w_diff'].values
    sa = df.loc[df['group'] == 'same array']['w_diff'].values
    oa = df.loc[df['group'] == 'other array']['w_diff'].values

    # KS tests
    p_se_sa, _ = ks_2samp(se, sa, alternative='less', mode='auto')
    p_se_oa, _ = ks_2samp(se, oa, alternative='less', mode='auto')
    p_sa_oa, _ = ks_2samp(sa, oa, alternative='less', mode='auto')

    oa_bins = [np.sum((oa >= intvals[i]) & (oa < intvals[i + 1])) for i in range(len(binz))]
    oa_bins = oa_bins / max(oa_bins)
    ax.bar(x=xbar, height=oa_bins, label='Other array')

    sa_bins = [np.sum((sa >= intvals[i]) & (sa < intvals[i + 1])) for i in range(len(binz))]
    sa_bins = sa_bins / max(sa_bins)
    ax.bar(x=xbar, height=sa_bins, label='Same array', alpha=.7)

    se_bins = [np.sum((se >= intvals[i]) & (se < intvals[i + 1])) for i in range(len(binz))]
    se_bins = se_bins / max(se_bins)
    ax.bar(x=xbar, height=se_bins,  label='Same electrode', alpha=.7)
    ax.set_xlabel(r'$w_{diff}$')
    ax.set_xticks([str(i / 10) for i in range(0, 6)])

    ax.annotate(f'Same elec to same array: p={round(p_se_sa,3)}', xy=(0.5, 0.5), xycoords=ax.transAxes)
    ax.annotate(f'Same elec to other array: p={round(p_se_oa,3)}', xy=(0.5, 0.4), xycoords=ax.transAxes)
    ax.annotate(f'Same array to other array: p={round(p_sa_oa,3)}', xy=(0.5, 0.3), xycoords=ax.transAxes)

    ax.legend()
    sns.despine()
    plt.show()




def KS_test(df):

    same_elec = df.loc[df['group'] == 'same elec']['w_diff'].values
    same_arr  = df.loc[df['group'] == 'same array']['w_diff'].values
    other_arr = df.loc[df['group'] == 'other array']['w_diff'].values

    p_se_sa, _ = ks_2samp(same_elec, same_arr, alternative='greater', mode='auto')
    p_se_oa, _ = ks_2samp(same_elec, other_arr, alternative='greater', mode='auto')
    p_sa_oa, _ = ks_2samp(same_arr, other_arr, alternative='greater', mode='auto')

    print(f'Same elec -  same arr: p={round(p_se_sa,3)}')
    print(f'Same elec - other arr: p={round(p_se_oa,3)}')
    print(f'Same arr  - other arr: p={round(p_sa_oa,3)}')
    
    return p_se_sa, p_se_oa, p_sa_oa


def visualize_traces(X_hat, dim, num_trials,):
    """
    Visualise PCA dimension traces over time.

    X_hat: df or np.array
        The projection of the data X on the chosen number of Principal Components (from sklearn model.transform())
    
    dim: int [2,3]
        whether to make a 2 or 3 dimensional plot.
    
    num_trials: number of trajectories to plot.
    """
    
    if dim == 2: # make 2d plot with first 2 PCs
        
        fig, ax = plt.subplots(1, figsize=(8, 6))
        for trial in np.arange(num_trials): 
            ax.plot(X_hat[trial][:, 0], X_hat[trial][:, 1], zorder=-1)
            cir = ax.scatter(X_hat[trial][:, 0][0], X_hat[trial][:, 1][0], color='g', marker='o')
            squ = ax.scatter(X_hat[trial][:, 0][-1], X_hat[trial][:, 1][-1], color='k', marker='s')
        ax.set_xlabel('PC 1', labelpad=10)
        ax.set_ylabel('PC 2', labelpad=10)
        ax.legend(handles=[cir, squ], labels=['start', 'end'])

    
    if dim == 3: # make 3d plot with first 3 PCs
        
        fig = plt.figure(figsize=(9, 7))
        ax = plt.axes(projection='3d')
        for trial in np.arange(num_trials): 
            ax.plot(X_hat[trial][:, 0], X_hat[trial][:, 1], X_hat[trial][:, 2], zorder=-1)
            cir = ax.scatter(X_hat[trial][:, 0][0], X_hat[trial][:, 1][0], X_hat[trial][:, 2][0], color='g', marker='o')
            squ = ax.scatter(X_hat[trial][:, 0][-1], X_hat[trial][:, 1][-1], X_hat[trial][:, 2][-1], color='k', marker='s')
        ax.set_xlabel('PC 1', labelpad=10)
        ax.set_ylabel('PC 2', labelpad=10)
        ax.set_zlabel('PC 3', labelpad=10)
        ax.legend(handles=[cir, squ], labels=['start', 'end'], loc='upper left')
        plt.tight_layout()



def sum_spikes_across_trials(td, spikes, verbose=False):
    '''
    Return array where each row represents the sum across trials of one neuron (Perich et al., 2021) Figure S1
    
    td: pd.dataframe
        trialdata
    spikes: str {'M1_spikes', 'PMd_spikes', 'both_spikes'}
    '''

    # take spike trains of all neurons for all trials
    spike_train_arr = td[spikes].values
    
    # get the maximum number of bins
    max_len = max([spike_train_arr[tr].shape[0] for tr in range(spike_train_arr.shape[0])])

    sum_spiketrains = np.zeros((spike_train_arr[0].shape[1], max_len))

    num_neurons = spike_train_arr[0].shape[1]
    num_trials = spike_train_arr.shape[0]

    for tr in range(num_trials):
        for n in range(num_neurons):

            if tr == 0:
                spikes_n_tr = spike_train_arr[tr][:, n]
                result = np.zeros(max_len)
                result[:spikes_n_tr.shape[0]:] = spikes_n_tr
                sum_spiketrains[n, :] = result
            else:
                spikes_n_tr = spike_train_arr[tr][:, n]
                result = np.zeros(max_len)
                result[:spikes_n_tr.shape[0]:] = spikes_n_tr
                sum_spiketrains[n, :] += result
    
    if verbose:
        fig, ax = plt.subplots(1, figsize=(8, 6))
        ax.set_title(f'Total spikes across trials for each neuron in {spikes}')
        sns.heatmap(sum_spiketrains[:, :50], xticklabels=False, yticklabels=False, cmap=cmap, ax=ax)
        ax.set_xlabel('Time')
        ax.set_ylabel('Neuron')
    
    return sum_spiketrains


def browse_maps(corr_emp_arr, corr_gen_arr, td, r_range, abs_val=False):
    ''' Browse through a set of matrices '''

    def view_map(i):

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
        fig.suptitle(f'Pairwise correlations with first {r_range[i]} PCs', y=1.02)

        axs[0].set_title(r'$\mathbf{X}_{emp}$')
        if abs_val == True:
            sns.heatmap(abs(corr_emp_arr[i]), cmap='mako', ax=axs[0])
        else:
            sns.heatmap(corr_emp_arr[i], cmap='coolwarm', ax=axs[0])
        axs[0].set_xlabel(r'pcs neuron $j$')
        axs[0].set_ylabel(r'pcs neuron $i$')

        axs[1].set_title(r'$\mathbf{X}_{rand}$')
        if abs_val == True: 
            sns.heatmap(abs(corr_gen_arr[i]), cmap='mako', ax=axs[1])
        else:
            sns.heatmap(corr_gen_arr[i], cmap='coolwarm', ax=axs[1])
        axs[1].set_xlabel(r'pcs neuron $j$')
        axs[1].set_ylabel(r'pcs neuron $i$')

        # Add white lines to indicate within and between array pc weights
        axs[0].vlines([0], 0, td.M1_spikes[0].shape[1], colors='w', lw=2)
        axs[0].hlines([0], 0, td.M1_spikes[0].shape[1], colors='w', lw=2)
        axs[0].vlines([td.M1_spikes[0].shape[1]], 0, td.M1_spikes[0].shape[1]+td.PMd_spikes[0].shape[1], colors='w', lw=2)
        axs[0].hlines([td.M1_spikes[0].shape[1]], 0, td.M1_spikes[0].shape[1]+td.PMd_spikes[0].shape[1], colors='w', lw=2)

        axs[1].vlines([0], 0, td.M1_spikes[0].shape[1], colors='w', lw=2)
        axs[1].hlines([0], 0, td.M1_spikes[0].shape[1], colors='w', lw=2)
        axs[1].vlines([td.M1_spikes[0].shape[1]], 0, td.M1_spikes[0].shape[1]+td.PMd_spikes[0].shape[1], colors='w', lw=2)
        axs[1].hlines([td.M1_spikes[0].shape[1]], 0, td.M1_spikes[0].shape[1]+td.PMd_spikes[0].shape[1], colors='w', lw=2)
        plt.show()
        
    interact(view_map, i=(0, corr_emp_arr.shape[0]-1))


def pairwise_corr_plot(corr_arr, spatial_dist_arr, i, r_range, td, date, monkey):
	'''

	Parameters
	----------
	corr_arr: matrix with all pairwise correlations

	spatial_dist_arr: spatial distance in same order

	r_range

	Returns
	-------
	'''
	fig = plt.figure(figsize=(15, 10))
	fig.suptitle(f'Pairwise correlations with first {r_range[i]} PCs \n Session {date} with monkey {monkey}', y=1.03, fontsize=15, color='grey')

	ax1 = plt.subplot2grid(shape = (2, 2), loc = (0, 0), colspan=1)
	ax2 = plt.subplot2grid(shape = (2, 2), loc = (0, 1), colspan=1)
	ax3 = plt.subplot2grid(shape = (2, 2), loc = (1, 0), colspan=2)

	# Plot the heatmaps
	ax1.set_title('Magnitude of correlation between neuron PCS')
	sns.heatmap(corr_arr[i], cmap='bwr', square=True, ax=ax1)
	ax1.set_xlabel(r'pcs neuron $j$')
	ax1.set_ylabel(r'pcs neuron $i$')

	# Add white lines to indicate within and between array pc weights
	ax1.vlines([0], 0, td.M1_spikes[0].shape[1], colors='w', lw=2)
	ax1.hlines([0], 0, td.M1_spikes[0].shape[1], colors='w', lw=2)
	ax1.vlines([td.M1_spikes[0].shape[1]], 0, td.M1_spikes[0].shape[1]+td.PMd_spikes[0].shape[1], colors='w', lw=2)
	ax1.hlines([td.M1_spikes[0].shape[1]], 0, td.M1_spikes[0].shape[1]+td.PMd_spikes[0].shape[1], colors='w', lw=2)

	ax2.set_title('Spatial distance between neurons')
	sns.heatmap(spatial_dist_arr, cmap=dist_cmap, square=True, ax=ax2)
	ax2.set_xlabel(r'pcs neuron $j$')
	ax2.set_ylabel(r'pcs neuron $i$')

	# Correlation plots
	spatial_dists_triu = spatial_dist_arr[np.triu_indices(corr_arr.shape[1], k=1)] # Take lower triangle (no repetition)
	corr_triu = corr_arr[i][np.triu_indices(corr_arr.shape[1], k=1)]
	nan_idx = np.argwhere(np.isnan(spatial_dists_triu))
	#corrs_squared = np.square(corr_triu)

	ax3.plot(spatial_dists_triu[~nan_idx], corr_triu[~nan_idx], '.', alpha=0.1, color='#1a54b3')
	ax3.set_title('Correlation values')
	ax3.set_xlabel('Spatial distance'), ax3.axhline(y=0, color='k', lw=1)
	ax3.set_ylabel(r'$\rho$')
	sns.despine()

	plt.tight_layout()

	df = pd.DataFrame({'corr': corr_triu, 'spatial dist': spatial_dists_triu}) # Make categories based on spatial distance
	df = df.dropna() # remove rows with np.nan in them (between vals)
	df['cat'] = pd.cut(df['spatial dist'], bins=[-0.1, 0.0001, 3, 6, 9, 16], labels=['0', '(0, 3]', '(3, 6]', '(6, 9]', '(9, 12]']) # Make categories

	pal = sns.cubehelix_palette(5, rot=-.25, light=.7, reverse=True)
	g = sns.FacetGrid(df, row='cat', hue='cat', aspect=9, height=.9, palette=pal)
	# Draw the densities in a few steps
	g.map(sns.kdeplot, 'corr', bw_adjust=.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
	g.map(sns.kdeplot, 'corr', clip_on=False, color='w', lw=2, bw_adjust=.5)
	g.map(plt.axhline, y=0, lw=2, clip_on=False)

	# Define and use a simple function to label the plot in axes coordinates
	def label(x, color, label):
	    ax = plt.gca()
	    ax.text(0, .2, label, fontweight='bold', color=color,
	            ha='left', va='center', transform=ax.transAxes, fontsize=14)

	g.map(label, 'corr')
	# Set the subplots to overlap
	g.fig.subplots_adjust(hspace=-.30)
	g.set_titles('') # Remove axes details that don't play well with overlap
	g.set(yticks=[])
	g.despine(bottom=True, left=True);
	plt.tight_layout();
