''' 
Date: 07.29.21 

This file contains the steps performed to preprocess the data before applying dimensionality reduction. 

'''
from pyaldata import * 
warnings.simplefilter('ignore')


def preprocess_data(session, path):
    '''
    Preprocess the data given an experimental session located at a path.

    Parameters
    ----------
    session: str
        the session to preprocess

    path: str
        data path

    Returns
    -------
    td: pandas dataframe
        the preprocessed data
    '''

    df = mat2dataframe(path + session, shift_idx_fields=True)

    # Combine bins 
    td = combine_time_bins(df, n_bins=3)

    # Remove low firing neurons
    td = remove_low_firing_neurons(td, signal='M1_spikes',  threshold=1)
    td = remove_low_firing_neurons(td, signal='PMd_spikes', threshold=1)

    # Sqrt transform neurons
    td = transform_signal(td, signals='M1_spikes',  transformations='sqrt')
    td = transform_signal(td, signals='PMd_spikes', transformations='sqrt')

    # Merge signals
    td = merge_signals(td, ['M1_spikes', 'PMd_spikes'], 'both_spikes')

    # Calculate firing rates from spikes, works on '_spikes' fields and automatically divides by bin_size
    td = add_firing_rates(td, 'smooth', std=0.05)

    # Select only baseline (BL) trials
    td = td.loc[td['epoch'] == 'BL']
    
    # Align trial data and restrict to interval - this removes quite a number of timepoints
    td = restrict_to_interval(td, 'idx_go_cue', end_point_name='idx_trial_end')
    
    return td