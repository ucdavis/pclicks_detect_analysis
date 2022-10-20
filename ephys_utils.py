# -*- coding: utf-8 -*-
"""
Set of functions to organize and manipulate spiking data 

@author: tanner stevenson
"""

import numpy as np
from scipy.stats import norm
import utils


def get_trial_spike_times(spike_times, trial_start_times):
    '''
    Organizes spike times by trial

    Parameters
    ----------
    spike_times : List of spike times
    trial_start_times : List of trial start times

    Returns
    -------
    A list of numpy arrays containing the spike times within each trial relative to the start of each trial 
    '''

    # make sure the spike times are a numpy array for logical comparisons
    spike_times = np.array(spike_times)
    trial_spike_times = []

    # we never push the last trial to the database
    for i in range(len(trial_start_times)-1):
        if i < len(trial_start_times)-2:
            spike_select = np.logical_and(spike_times > trial_start_times[i], spike_times < trial_start_times[i+1])
        else:
            spike_select = spike_times > trial_start_times[i]

        trial_spike_times.append(spike_times[spike_select] - trial_start_times[i])

    return trial_spike_times


def get_binned_spike_counts(spike_times, start_time=0, end_time=np.inf, bin_width=5e-3):
    '''
    Gets binned spike counts for the given spike times between the start and end time points with the given bin width

    Parameters
    ----------
    spike_times : List of spike times
    start_time : (optional) The start time of the bins. The default is 0.
    end_time : (optional) The end time of the bins. The default is one bin width beyond the last spike time
    bin_width : (optional) The width of the bins. The default is 5e-3.

    Returns
    -------
    counts : An array of spike counts within each bin
    bin_edges : The bin edges. Will have one more element than the counts
    '''

    if np.isinf(end_time):
        end_time = utils.convert_to_multiple(end_time-start_time, bin_width)

    bin_edges = np.arange(start_time, end_time, bin_width)
    counts, _ = np.histogram(spike_times, bin_edges)

    return counts, bin_edges


def get_filter_kernel(width, filter_type='half_gauss', bin_width=5e-5):
    '''
    Gets a dictionary with entries that contain information used by filtering routines

    Parameters
    ----------
    width : The width of the filter
    filter_type : (optional) The type of filter. Acceptable values: 'avg', 'causal_avg', 'gauss', 'half_gauss', 'exp', and 'none'. 
        The default is 'half_gauss'.
    bin_width : (optional) The width of the bin. The default is 5e-5.

    Returns
    -------
    A dictionary with entries:
        type : the filter type
        weights : the weights used when filtering
        bin_width : the width of the filter bin
        center_idx : the index of the center of the filter
    '''

    # average filter with window centered on the current bin
    if filter_type == 'avg':
        window_limit = utils.convert_to_multiple(width/2, bin_width)
        x = np.arange(-window_limit, window_limit, bin_width)
        weights = np.ones_like(x)

    # causal average filter that only considers prior signal with equal weights
    elif filter_type == 'causal_avg':
        # set the window limit to a multiple of bin width
        window_limit = utils.convert_to_multiple(width, bin_width)
        x = np.arange(0, window_limit, bin_width)
        weights = np.ones_like(x)

    # gaussian filter with max centered on current bin
    elif filter_type == 'gauss':
        # set the window limit to a multiple of bin width
        window_limit = utils.convert_to_multiple(width/2, bin_width)
        x = np.arange(-window_limit, window_limit, bin_width)
        weights = norm.pdf(x, 0, window_limit/4)

    # causal filter that only considers prior signal with gaussian weights
    elif filter_type == 'half_gauss':
        # set the window limit to a multiple of bin width
        window_limit = utils.convert_to_multiple(width, bin_width)
        x = np.arange(0, window_limit, bin_width)
        weights = norm.pdf(x, 0, window_limit/4)

    # causal filter that only considers prior signal with exponentially decaying weights
    elif filter_type == 'exp':
        # set the window limit to a multiple of bin width
        window_limit = utils.convert_to_multiple(width, bin_width)
        x = np.arange(0, window_limit, bin_width)
        weights = np.exp(-x*4/window_limit)

    # no filter
    elif filter_type == 'none':
        x = 0
        weights = 1

    else:
        raise ValueError('Invalid filter type. Acceptable types: avg, causal_avg, gauss, half_gauss, exp, and none')

    return {'type': filter_type,
            'weights': weights/np.sum(weights),  # normalize sum to one
            'bin_width': bin_width,
            'center_idx': np.where(x == 0)[0][0]}


def get_smoothed_firing_rate(spike_times, kernel, start_time=0, end_time=np.inf):
    '''
    Will calculate a smoothed firing rate based on the spike times between the given start and end times

    Parameters
    ----------
    spike_times : List of spike times
    kernel : A kernel dictionary from get_filter_kernel
    start_time : (optional) The start time of smoothed signal. The default is 0.
    end_time : (optional) The end time of the smoothed signal. The default is the last spike time

    Returns
    -------
    signal : A smoothed firing rate with points separated by the bin width specified by the kernel
    time : The time values corresponding to the signal values
    '''

    bin_width = kernel['bin_width']

    if np.isinf(end_time):
        end_time = utils.convert_to_multiple(end_time-start_time, bin_width)

    # compute buffers around the start and end times to include spikes that should be included in the filter
    # shift them by half a bin width to make the resulting time have a value at 0
    pre_buff = (len(kernel['weights']) - kernel['center_idx']) * bin_width + bin_width/2
    post_buff = (kernel['center_idx'] - 1) * bin_width + bin_width/2

    # compute signal and smooth it with a filter
    signal, bin_edges = get_binned_spike_counts(spike_times, start_time-pre_buff, end_time+post_buff, bin_width)
    signal = signal/bin_width
    signal = np.convolve(signal, kernel['weights'])

    # remove extra bins created from filtering
    filter_pre_cutoff = len(kernel['weights'])
    filter_post_cutoff = len(kernel['weights']) - 1
    signal = signal[filter_pre_cutoff:-filter_post_cutoff]

    # get time as the middle of each bin
    time = (bin_edges[1:] + bin_edges[:-1]) / 2

    return signal, time


def get_psth(spike_times, align_times, kernel, window, mask_bounds=None):
    '''
    Will calculate a peri-stimulus time histogram (PSTH) of the average firing rate aligned to the specified alignment points

    Parameters
    ----------
    spike_times : A list of spike times, either a single list or a list of spike times separated by trial (N trials)
    align_times : A list of N event times to align the firing rates.
        Can be a single list or a list of lists if there are multiple (K*) alignment points per trial.
        Time is relative to start of the trial
    kernel : The smoothing filter kernel
    window : The window (pre, post) around the alignment points that define the bounds of the psth. 
        Time is relative to alignment point
    mask_bounds : (optional) The boundaries (pre, post) per trial past which any signal should be removed before averaging.
        Either a Nx2 matrix of boundaries or a list of N K*x2 matrices if there are K* alignment points per trial.
        Time is relative to start of the trial.

    Returns
    -------
    A dictionary with entries:
        signal_avg : The smoothed average signal
        signal_se : The signal standard error
        time : The time vector corresponding to the signal
        all_signals : All smoothed signals
    '''

    ## Handle multiple forms of inputs and check dimensions ##

    # make single vector of spike times represent a single 'trial'
    if utils.is_scalar(spike_times[0]):
        spike_times = [spike_times]

    n_trials = len(spike_times)

    # make single vector of alignment times represent multiple points within a single 'trial'
    if utils.is_scalar(align_times[0]):
        align_times = [align_times]

    # check the number of trials matches up
    if len(align_times) != n_trials:
        raise ValueError('The number of alignment points ({0}) does not match the number of trials ({1})'.format(
            len(align_times), n_trials))

    # check the mask bounds
    has_mask = not mask_bounds is None
    if has_mask:
        # make single matrix of mask bounds be for multiple alignment points in a single 'trial'
        if isinstance(mask_bounds, np.ndarray):
            mask_bounds = [mask_bounds]

        # check the number of trials matches up
        if len(mask_bounds) != n_trials:
            raise ValueError('The number of mask bounds ({0}) does not match the number of trials ({1})'.format(
                len(mask_bounds), n_trials))

    ## Perform aligning and smoothing ##

    _, time = get_smoothed_firing_rate([], kernel, window[0], window[1])

    if has_mask:
        time_bin_edges = np.append(time - kernel['bin_width']/2, time[-1] + kernel['bin_width']/2)

    all_signals = []

    for i in range(n_trials):
        trial_spike_times = np.array(spike_times[i])
        trial_align_times = align_times[i]

        if has_mask:
            trial_mask_bounds = mask_bounds[i]

            # make sure number of mask bounds is the same as number of alignment points
            if not np.shape(trial_mask_bounds)[0] != len(trial_align_times):
                raise ValueError('The number of trial mask bounds ({0}) does not match the number of alignment points ({1}) in trial {2}'.format(
                    np.shape(trial_mask_bounds)[0], len(trial_align_times), i))

        for j, align_ts in enumerate(trial_align_times):
            offset_ts = trial_spike_times - align_ts
            signal, _ = get_smoothed_firing_rate(offset_ts, kernel, window[0], window[1])

            # mask the signal
            if has_mask:
                align_mask = trial_mask_bounds[j, :]

                # find mask indices
                mask_start = align_mask[0] - align_ts
                mask_end = align_mask[1] - align_ts
                mask_start_idx = np.argmax(time_bin_edges > mask_start)
                mask_end_idx = np.argmax(time_bin_edges > mask_end) - 1

                # mask with nans
                if mask_start_idx > 0:
                    signal[0:mask_start_idx] = np.nan
                if mask_end_idx < len(signal):
                    signal[mask_end_idx:] = np.nan

            all_signals = all_signals.append(signal)

    # convert all signals list to matrix
    all_signals = np.array(all_signals)
    # compute average and standard error
    return {'signal_avg': np.nanmean(all_signals, axis=0),
            'signal_se': utils.stderr(all_signals),
            'time': time,
            'all_signals': all_signals}
