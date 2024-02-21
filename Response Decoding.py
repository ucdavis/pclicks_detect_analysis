# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:14:54 2023

@author: tanne
"""

# %% Imports
import sys
import os
from os import path
sys.path.append(path.join(path.dirname(path.abspath(__file__)), '..'))

from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
import sklearn.model_selection as skl_ms
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy.random import default_rng
import numpy as np
import pandas as pd
from sys_neuro_tools import ephys_utils, plot_utils
import pyutils.utils as utils
import hankslab_db.pclicksdetect_db as db
from hankslab_db import db_access
import time
import warnings
import pickle

rng = default_rng()

# %% Declare decoding options and load any saved data if it exists

data_dir = path.join(path.dirname(path.abspath(__file__)), 'save_data')
if not path.exists(data_dir):
    os.mkdir(data_dir)

subj_ids = [78,94]
# get session ids for the ephys recordings
# sess_ids = db_access.get_subj_unit_sess_ids(subj_ids)
# # only look at a subset of sessions with decent amounts of units in all regions
# sess_select = {78: np.concatenate((np.arange(11), np.arange(12,15))),
#                94: np.arange(13)}
# # select out the session ids and flatten into single list
# sess_ids = [i for sub, id_sel in sess_select.items() for i in np.array(sess_ids[sub])[id_sel].tolist()]

sess_ids = [47709, 48447]

# declare structures used during analysis
kernel = ephys_utils.get_filter_kernel(width=0.3, bin_width=0.05)  # defaults to causal half gaussian with a width of 0.2 and bin width of 5e-3

n_shuffle_reps = 100
n_folds = 10

# delays are intentionally created in reverse order for indexing purposes
delays = np.arange(-kernel['bin_width'], -0.8-kernel['bin_width'], -kernel['bin_width'])
region_sel = ['All', 'FOF', 'ADS', 'CC']
single_units_only = False

# declare classifier and cross-validator
#classifier = svm.LinearSVC(dual=True, max_iter=10000)
#classifier = svm.NuSVC()
classifier = lda()
cv = skl_ms.StratifiedKFold(n_folds, shuffle=True)

save_filename = 'resp_decoding_LDA_all_units.pkl'

save_file_path = path.join(data_dir, save_filename)
use_save_data = False
if path.exists(save_file_path):
    with open(save_file_path, 'rb') as f:
        save_data = pickle.load(f)

    use_save_data = True

    # make sure save data has the same options
    if save_data['single_units_only'] != single_units_only:
        print('WARNING: save data at {} does not have the same single units flag. current: {}. save data: {}. Cannot use saved data.'
              .format(save_file_path, single_units_only, save_data['single_units_only']))
        use_save_data = False

    if use_save_data and type(save_data['classifier']) != type(classifier):
        print('WARNING: save data at {} does not have the same classifier. current: {}. save data: {}. Cannot use saved data.'
              .format(save_file_path, type(classifier), type(save_data['classifier'])))
        use_save_data = False

    if use_save_data and not (save_data['kernel']['type'] == kernel['type'] and
                              save_data['kernel']['bin_width'] == kernel['bin_width'] and
                              np.array_equal(save_data['kernel']['weights'], kernel['weights'])):
        print('WARNING: save data at {} does not have the same kernel. current: {}. save data: {}. Cannot use saved data.'
              .format(save_file_path, kernel, save_data['kernel']))
        use_save_data = False

    # temporary errors. Can be addressed if needed, but is much more work
    if use_save_data and not np.array_equal(save_data['delays'], delays):
        print('WARNING: save data at {} does not have the same delays. current: {}. save data: {}. Cannot use saved data.'
              .format(save_file_path, delays, save_data['delays']))
        use_save_data = False

    if use_save_data and not (save_data['n_shuffle_reps'] == n_shuffle_reps and
                              save_data['n_folds'] == n_folds):
        print('WARNING: save data at {} does not have the same repeats/folds. current: {}/{}. save data: {}/{}. Cannot use saved data.'
              .format(save_file_path, n_shuffle_reps, n_folds, save_data['n_shuffle_reps'], save_data['n_folds']))
        use_save_data = False

    if use_save_data:

        unit_sel_dict = save_data['unit_sel_dict']
        accy_dict = save_data['accy_dict']
        num_trials_dict = save_data['num_trials_dict']

        sess_select = [sess_id in save_data['completed_sess_ids'] for sess_id in sess_ids]
        if not all(sess_select):
            print('WARNING: save data at {} is missing session ids: {}. Adding them into saved data..'
                  .format(save_file_path, sess_ids[sess_select]))

            sess_ids_to_run = sess_ids[sess_select]
            for sess_id in sess_ids_to_run:
                unit_sel_dict[sess_id] = {'FOF': {}, 'ADS': {}, 'CC': {}, 'Single': {}}
                accy_dict[sess_id] = {sel: np.zeros((n_shuffle_reps*n_folds, len(delays))) for sel in region_sel}
                num_trials_dict[sess_id] = {d: [] for d in delays}


if not use_save_data:
    sess_ids_to_run = sess_ids
    save_data = {}
    unit_sel_dict = {sess_id: {'FOF': {}, 'ADS': {}, 'CC': {}, 'Single': {}} for sess_id in sess_ids}
    accy_dict = {sess_id: {sel: np.zeros((n_shuffle_reps*n_folds, len(delays))) for sel in region_sel} for sess_id in sess_ids}
    num_trials_dict = {sess_id: {d: [] for d in delays} for sess_id in sess_ids}

# %% Load the data

loc_db = db.LocalDB_PClicksDetect()

unit_data = loc_db.get_sess_unit_data(sess_ids_to_run)
sess_data = loc_db.get_behavior_data(sess_ids_to_run)

sess_unit_info = unit_data.groupby(['subjid','sessid']).agg(
    single_units=('single_unit', 'sum'), all_units=('single_unit', 'count')
    ).sort_values(['subjid','single_units'], ascending=False).reset_index()

# %% Get smoothed firing rates during the poke period

fr_dict = {sess_id: {} for sess_id in sess_ids_to_run}
bounds_dict = {sess_id: {} for sess_id in sess_ids_to_run}

for sess_id in sess_ids_to_run:

    sess_trials = sess_data[sess_data['sessid'] == sess_id].reset_index(drop=True)
    n_trials = len(sess_trials)

    all_units = unit_data[(unit_data['sessid'] == sess_id) & (unit_data['number_spikes'] > n_trials)].reset_index(drop=True)
    unit_sel_dict[sess_id]['Single'] = all_units['single_unit'] == 1
    unit_sel_dict[sess_id]['FOF'] = all_units['region'] == 'FOF'
    unit_sel_dict[sess_id]['ADS'] = all_units['region'] == 'ADS'
    unit_sel_dict[sess_id]['CC'] = all_units['region'] == 'CC'

    # get trials where the animal makes a response
    # ignore FAs that are too fast to be driven by the stimulus
    resp_select = (sess_trials['hit'] == 1) | ((sess_trials['FA'] == 1) & (sess_trials['cpoke_dur'] > 0.5)) & (sess_trials['valid'])

    # compute all timestamps with relation to the response since that is what we are interested in decoding
    trial_start_ts = all_units['trial_start_timestamps'].iloc[0][0:-1]
    # note: last trial start is given by the ttl pulse at the start of the trial that is incomplete at the end of the session so ignore it
    abs_cpoke_out = trial_start_ts + sess_trials['cpoke_out'].to_numpy() # cpoke_out relative to start of ephys, this will be our effective 0 point

    rel_cpoke_start = sess_trials['cpoke_start'].to_numpy() - sess_trials['cpoke_out'].to_numpy()
    rel_cpoke_start = rel_cpoke_start.reshape(-1,1)
    rel_poke_bounds = np.concatenate((rel_cpoke_start, np.zeros_like(rel_cpoke_start)), axis=1)
    bounds_dict[sess_id] = rel_poke_bounds[resp_select,:]

    # compute the smoothed firing rate matrices
    fr_dict[sess_id] = ephys_utils.get_fr_matrix_by_trial(all_units['spike_timestamps'], abs_cpoke_out, kernel, rel_poke_bounds, resp_select)


# %% Decode whether there will be a movement in a certain delay or not

# have a trial length buffer that the negative trial time bin has to be greater than
trial_len_buff = 0.1

for s, sess_id in enumerate(sess_ids_to_run):
    sess_start = time.perf_counter()
    print('Starting session {0}'.format(sess_id))

    poke_bounds = bounds_dict[sess_id]
    poke_dur = poke_bounds[:,1] - poke_bounds[:,0]

    for i, delay in enumerate(delays):
        delay_start = time.perf_counter()

        # select trials with poke durations longer than the delay to split up trials evenly
        delay_trial_select = poke_dur >= (-delay + trial_len_buff)
        n_delay_trials = sum(delay_trial_select)
        delay_trial_idxs = np.arange(n_delay_trials)
        delay_poke_durs = poke_dur[delay_trial_select]

        num_trials_dict[sess_id][delay] = n_delay_trials

        shuffle_start = time.perf_counter()
        for j in range(n_shuffle_reps):
            # randomly assign trials to each group
            # positive trials are those where the time bin is correct, negative are those where the bin is incorrect
            pos_trial_idxs = np.sort(rng.choice(delay_trial_idxs, np.ceil(n_delay_trials/2).astype(int), replace=False))
            neg_trial_idxs = delay_trial_idxs[[idx not in pos_trial_idxs for idx in delay_trial_idxs]]

            # randomly align the negative trials so the time bin used to train the decoder
            # is at a random point in the trial before the delay
            neg_trial_remainders = delay_poke_durs[neg_trial_idxs] + delay
            neg_trial_offsets = np.maximum(rng.random(len(neg_trial_idxs)) * neg_trial_remainders, trial_len_buff)
            # compute delay index for trial offsets
            neg_trial_offset_idxs = np.ceil(neg_trial_offsets / kernel['bin_width']).astype(int)

            # create classifier labels. pos are 1, neg are -1
            labels = np.zeros(n_delay_trials) - 1
            labels[pos_trial_idxs] = 1

            # reuse the same shuffling for both selections of units
            for region in region_sel:

                # filter firing rate matrices by trials and by units
                delay_trial_fr = fr_dict[sess_id].loc[delay_trial_select].reset_index(drop=True)

                if single_units_only:
                    unit_sel = unit_sel_dict[sess_id]['Single']
                else:
                    unit_sel = np.array([True for i in range(len(unit_sel_dict[sess_id]['Single']))])

                if region != 'All':
                    unit_sel = unit_sel & unit_sel_dict[sess_id][region]

                if sum(unit_sel) == 0:
                    print('WARNING: session {} has no units in {} region(s)'.format(sess_id, region))
                    continue

                delay_trial_fr = delay_trial_fr.apply(lambda x: x[:,unit_sel])

                # get unit activity to train classifier, trials in rows, units in columns
                unit_act = np.zeros((n_delay_trials, np.size(delay_trial_fr.iloc[0], 1)))
                # subtract 2 from index because movement happens in the last bin, which is indexed as -1 and i starts from 0
                # so when decoding whether the movement happens at a delay of one bin width, we need to start at index -2
                unit_act[pos_trial_idxs,:] = np.vstack(delay_trial_fr.iloc[pos_trial_idxs].apply(lambda x: x[-i-2,:]))
                unit_act[neg_trial_idxs,:] = np.vstack([delay_trial_fr.iloc[idx][-i-2-neg_trial_offset_idxs[k],:] for k, idx in enumerate(neg_trial_idxs)])

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scores = skl_ms.cross_val_score(classifier, unit_act, labels, groups=labels, cv=cv)

                accy_dict[sess_id][region][j*n_folds:(j+1)*n_folds,i] = scores

            if (j+1) % 10 == 0:
                print('    Completed shuffle {}/{} for delay {} in {:.1f} s'.format(j+1, n_shuffle_reps, delay, time.perf_counter()-shuffle_start))
                shuffle_start = time.perf_counter()

        print('  Completed delay {0} in {1:.1f} s'.format(delay, time.perf_counter()-delay_start))

    print('Completed session {0} in {3:.1f} s. {1}/{2} sessions'.format(sess_id, s+1, len(sess_ids), time.perf_counter()-sess_start))

# %% Save Decoding Results
# if all decoding was completed, save all session ids
if s == len(sess_ids_to_run)-1 and j == n_shuffle_reps-1:
    completed_sess_ids = sess_ids_to_run
else:
    completed_sess_ids = sess_ids_to_run[:s]

# if there was already save data, combine the completed session ids
if len(save_data) > 0:
    prior_ids = save_data['completed_sess_ids']
    prior_ids.extend(completed_sess_ids)
    completed_sess_ids = prior_ids

save_data = {'completed_sess_ids': completed_sess_ids, 'accy_dict': accy_dict,
             'num_trials_dict': num_trials_dict, 'unit_sel_dict': unit_sel_dict,
             'single_units_only': single_units_only, 'delays': delays,
             'kernel': kernel, 'n_shuffle_reps': n_shuffle_reps, 'n_folds': n_folds,
             'classifier': classifier, 'cv': cv}

with open(save_file_path, 'wb') as f:
    pickle.dump(save_data, f, protocol=5)

# %% Plot decoding accuracy

for sess_id in sess_ids:
    unit_info = sess_unit_info.query('sessid=={}'.format(sess_id))
    subj_id = unit_info['subjid'].iloc[0]

    if single_units_only:
        unit_type = 'Single'
        n_units = sum(unit_sel_dict[sess_id]['Single'])
    else:
        unit_type = 'All'
        n_units = len(unit_sel_dict[sess_id]['Single'])

    fig, axs = plt.subplots(1,2, sharex=True, sharey=True, figsize=(10,4), layout='constrained')

    fig.suptitle('Rat {} Session {} {} Units'.format(subj_id, sess_id, unit_type))
    # plot all regions first
    ax = axs[0]
    ax.axhline(0.5, dashes=[4, 4], c='k', lw=1)

    sess_accy = accy_dict[sess_id]['All']
    avg_accy = np.mean(sess_accy, axis=0)
    #err_accy = utils.stderr(sess_accy, axis=0)
    err_accy = np.std(sess_accy, axis=0)

    plot_utils.plot_shaded_error(delays, avg_accy, err_accy, ax=ax)

    ax.set_xlabel('Time from response (s)')
    ax.set_ylabel('Accuracy')
    ax.set_xlim([delays[-1],0])
    ax.set_ylim([0.3,1])
    ax.set_title('All Regions (n={:.0f} units)'.format(n_units))

    # plot number of trials at each delay
    ax = ax.twinx()
    color = 'tab:red'
    ax.plot(delays, [num_trials_dict[sess_id][d] for d in delays], color=color)
    ax.set_ylabel('Trial Counts', color=color)
    ax.set_ylim([0,500])
    ax.tick_params(axis='y', labelcolor=color)

    ax = axs[1]
    ax.axhline(0.5, dashes=[4, 4], c='k', lw=1)
    ax.set_title('By Region')
    ax.set_xlabel('Time from response (s)')
    ax.yaxis.set_tick_params(labelleft=True)

    for region in region_sel:

        if region == 'All':
            continue

        if single_units_only:
            n_units = sum((unit_sel_dict[sess_id][region]) & (unit_sel_dict[sess_id]['Single']))
        else:
            n_units = sum(unit_sel_dict[sess_id][region])

        sess_accy = accy_dict[sess_id][region]
        avg_accy = np.mean(sess_accy, axis=0)
        #err_accy = utils.stderr(sess_accy, axis=0)
        err_accy = np.std(sess_accy, axis=0)

        plot_utils.plot_shaded_error(delays, avg_accy, err_accy, ax=ax, label='{} (n={:.0f})'.format(region, n_units))

    ax.legend()

# %% Get control decoding accuracy if units had firing rates that varied linearly with time in trial

n_units = 200
n_trials = 400
base_fr = rng.random(n_units)*5
slopes = rng.random(n_units)*10

# get trial lengths
fixed_time = 0.5 # pre-change
resp_mean = 1.5
resp_max = 3.5
trial_lens = rng.exponential(resp_mean, (n_trials, 10))
trial_len_idxs = np.argmax(trial_lens < resp_max, axis=1)
trial_lens = fixed_time + trial_lens[np.arange(n_trials), trial_len_idxs]

frs_by_trial = pd.Series([base_fr[None,:] + slopes[None,:]*np.arange(0, t, kernel['bin_width'])[:,None] for t in trial_lens])

accy = np.zeros((n_shuffle_reps*n_folds, len(delays)))
num_trials = {d: [] for d in delays}

for i, delay in enumerate(delays):
    delay_start = time.perf_counter()

    # select trials with poke durations longer than the delay to split up trials evenly
    delay_trial_select = trial_lens >= (-delay + trial_len_buff)
    n_delay_trials = sum(delay_trial_select)
    delay_trial_idxs = np.arange(n_delay_trials)
    delay_poke_durs = trial_lens[delay_trial_select]

    delay_trial_fr = frs_by_trial.loc[delay_trial_select].reset_index(drop=True)

    num_trials[delay] = n_delay_trials

    shuffle_start = time.perf_counter()

    for j in range(n_shuffle_reps):
        # randomly assign trials to each group
        # positive trials are those where the time bin is correct, negative are those where the bin is incorrect
        pos_trial_idxs = np.sort(rng.choice(delay_trial_idxs, np.ceil(n_delay_trials/2).astype(int), replace=False))
        neg_trial_idxs = delay_trial_idxs[[idx not in pos_trial_idxs for idx in delay_trial_idxs]]

        # randomly align the negative trials so the time bin used to train the decoder
        # is at a random point in the trial before the delay
        neg_trial_remainders = delay_poke_durs[neg_trial_idxs] + delay
        neg_trial_offsets = np.maximum(rng.random(len(neg_trial_idxs)) * neg_trial_remainders, trial_len_buff)
        # compute delay index for trial offsets
        neg_trial_offset_idxs = np.ceil(neg_trial_offsets / kernel['bin_width']).astype(int)

        # create classifier labels. pos are 1, neg are -1
        labels = np.zeros(n_delay_trials) - 1
        labels[pos_trial_idxs] = 1

        # get unit activity to train classifier, trials in rows, units in columns
        unit_act = np.zeros((n_delay_trials, np.size(delay_trial_fr.iloc[0], 1)))
        # subtract 2 from index because movement happens in the last bin, which is indexed as -1 and i starts from 0
        # so when decoding whether the movement happens at a delay of one bin width, we need to start at index -2
        unit_act[pos_trial_idxs,:] = np.vstack(delay_trial_fr.iloc[pos_trial_idxs].apply(lambda x: x[-i-1,:]))
        unit_act[neg_trial_idxs,:] = np.vstack([delay_trial_fr.iloc[idx][-i-1-neg_trial_offset_idxs[k],:] for k, idx in enumerate(neg_trial_idxs)])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = skl_ms.cross_val_score(classifier, unit_act, labels, groups=labels, cv=cv)

        accy[j*n_folds:(j+1)*n_folds,i] = scores

        if (j+1) % 10 == 0:
            print('    Completed shuffle {}/{} for delay {} in {:.1f} s'.format(j+1, n_shuffle_reps, delay, time.perf_counter()-shuffle_start))
            shuffle_start = time.perf_counter()

    print('  Completed delay {0} in {1:.1f} s'.format(delay, time.perf_counter()-delay_start))

# %%
save_filename = 'control_resp_decoding_time_neurons.pkl'

save_data = {'accy': accy, 'num_trials': num_trials, 'delays': delays,
             'base_fr': base_fr, 'slopes': slopes, 'trial_lens': trial_lens,
             'kernel': kernel, 'n_shuffle_reps': n_shuffle_reps, 'n_folds': n_folds,
             'classifier': classifier, 'cv': cv}

with open(path.join(data_dir, save_filename), 'wb') as f:
    pickle.dump(save_data, f, protocol=5)

# %%
fig, ax = plt.subplots(1,1, constrained_layout=True, figsize=(6,4))

fig.suptitle('Decoding response with units that encode time (n={} units)'.format(n_units))
ax.axhline(0.5, dashes=[4, 4], c='k', lw=1)

avg_accy = np.mean(accy, axis=0)
#err_accy = utils.stderr(sess_accy, axis=0)
err_accy = np.std(accy, axis=0)

plot_utils.plot_shaded_error(delays, avg_accy, err_accy, ax=ax)

ax.set_xlabel('Time from response (s)')
ax.set_ylabel('Accuracy')
ax.set_xlim([delays[-1],0])
ax.set_ylim([0.3,1])

# plot number of trials at each delay
ax = ax.twinx()
color = 'tab:red'
ax.plot(delays, [num_trials[d] for d in delays], color=color)
ax.set_ylabel('Trial Counts', color=color)
ax.set_ylim([0,500])
ax.tick_params(axis='y', labelcolor=color)

