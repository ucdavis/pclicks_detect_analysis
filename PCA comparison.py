# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 12:10:02 2023

@author: tanne
"""

# %% Imports
import sys
from os import path
sys.path.append(path.join(path.dirname(path.abspath(__file__)), '..'))

from sklearn.decomposition import PCA
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from numpy.random import default_rng
import numpy as np
from sys_neuro_tools import ephys_utils
import hankslab_db.pclicksdetect_db as db
import time

rng = default_rng(1)

# %% Load the data

loc_db = db.LocalDB_PClicksDetect()
# subj_ids = loc_db.get_protocol_subject_ids() # will get all subjects recorded during this protocol
# for now lets just use these ones
subj_ids = [78,94]  # , 94

# get all unit and session data for these subjects
# this will take a while (~60 mins) the first time
unit_data = loc_db.get_subj_unit_data(subj_ids)
sess_data = loc_db.get_subj_behavior_data(subj_ids)

# %% Declare reusable plotting methods
# declare separate colors for different task epochs
epoch_colors = {'prepoke': 'khaki', 'poke': 'skyblue', 'postpoke': 'tomato'}
poke_start_shape = 'o'
poke_end_shape = 'v'

# define reusable method for plotting
def plot_pc_activity(ax, pca, trial_act, trial_start_ts, cpoke_start_ts, cpoke_end_ts, dt, title='',
                     color_type='epoch', colors=epoch_colors, norm_trial_idxs=[], x_pc=1, y_pc=2):

    trial_act.reset_index(drop=True, inplace=True)
    for i in range(len(trial_start_ts)):
        activity_pc = pca.transform(trial_act[i])
        cpoke_start_idx = np.ceil((cpoke_start_ts[i] - trial_start_ts[i])/dt).astype(int)
        cpoke_end_idx = np.floor((cpoke_end_ts[i] - trial_start_ts[i])/dt).astype(int)
        # plot activity lines
        if color_type == 'epoch':
            ax.plot(activity_pc[:cpoke_start_idx, x_pc-1], activity_pc[:cpoke_start_idx, y_pc-1],
                    color=colors['prepoke'], alpha=0.4, label='_')
            ax.plot(activity_pc[cpoke_start_idx:cpoke_end_idx, x_pc-1], activity_pc[cpoke_start_idx:cpoke_end_idx, y_pc-1],
                    color=colors['poke'], alpha=0.4, label='_')
            ax.plot(activity_pc[cpoke_end_idx:, x_pc-1], activity_pc[cpoke_end_idx:, y_pc-1],
                    color=colors['postpoke'], alpha=0.4, label='_')
        elif color_type == 'trial':
            ax.plot(activity_pc[:, x_pc-1], activity_pc[:, y_pc-1],
                    color=colors(norm_trial_idxs[i]), alpha=0.4, label='_')
        # plot transitions as points
        ax.plot(activity_pc[cpoke_start_idx, x_pc-1], activity_pc[cpoke_start_idx, y_pc-1], poke_start_shape,
                color='black', alpha=0.7, label='_')
        ax.plot(activity_pc[cpoke_end_idx, x_pc-1], activity_pc[cpoke_end_idx, y_pc-1], poke_end_shape,
                color='black', alpha=0.7, label='_')

        ax.set_xlabel('PC {}'.format(x_pc))
        ax.set_ylabel('PC {}'.format(y_pc))
        ax.set_title(title)

def compare_pc_act(sess_id, sess_unit_info, trial_select_dict, pca_dict, fr_dict, trial_event_dict, kernel, n_plot_trials):
    # select random individual trials to plot in PC space
    hit_idxs_to_plot = np.sort(rng.choice(trial_select_dict[sess_id]['hit_idxs'], n_plot_trials, replace=False))
    fa_idxs_to_plot = np.sort(rng.choice(trial_select_dict[sess_id]['fa_idxs'], n_plot_trials, replace=False))

    hit_trials_plot_sel = [idx in hit_idxs_to_plot for idx in trial_select_dict[sess_id]['hit_idxs']]
    fa_trials_plot_sel = [idx in fa_idxs_to_plot for idx in trial_select_dict[sess_id]['fa_idxs']]

    unit_info = sess_unit_info.query('sessid=={}'.format(sess_id))
    subj_id = unit_info['subjid'].iloc[0]
    title_prefix = 'Rat {} Session {}'.format(subj_id, sess_id)

    # first color by epoch in trial
    # Create 2x2 figure - hits & fas in different rows, poke and trial in different columns
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)

    # first plot single units, then plot all units
    plot_pc_activity(axs[0,0], pca_dict[sess_id]['single']['poke'],
                     fr_dict[sess_id]['single']['poke_hits'][hit_trials_plot_sel],
                     trial_event_dict[sess_id]['poke']['trial_bounds'][hit_idxs_to_plot,0],
                     trial_event_dict[sess_id]['poke']['cpoke_start'][hit_idxs_to_plot],
                     trial_event_dict[sess_id]['poke']['cpoke_out'][hit_idxs_to_plot],
                     kernel['bin_width'], title='Hits - poke')

    plot_pc_activity(axs[1,0], pca_dict[sess_id]['single']['poke'],
                     fr_dict[sess_id]['single']['poke_fas'][fa_trials_plot_sel],
                     trial_event_dict[sess_id]['poke']['trial_bounds'][fa_idxs_to_plot,0],
                     trial_event_dict[sess_id]['poke']['cpoke_start'][fa_idxs_to_plot],
                     trial_event_dict[sess_id]['poke']['cpoke_out'][fa_idxs_to_plot],
                     kernel['bin_width'], title='FAs - poke')

    plot_pc_activity(axs[0,1], pca_dict[sess_id]['single']['extended'],
                     fr_dict[sess_id]['single']['extended_hits'][hit_trials_plot_sel],
                     trial_event_dict[sess_id]['extended']['trial_bounds'][hit_idxs_to_plot,0],
                     trial_event_dict[sess_id]['extended']['cpoke_start'][hit_idxs_to_plot],
                     trial_event_dict[sess_id]['extended']['cpoke_out'][hit_idxs_to_plot],
                     kernel['bin_width'], title='Hits - extended')

    plot_pc_activity(axs[1,1], pca_dict[sess_id]['single']['extended'],
                     fr_dict[sess_id]['single']['extended_fas'][fa_trials_plot_sel],
                     trial_event_dict[sess_id]['extended']['trial_bounds'][fa_idxs_to_plot,0],
                     trial_event_dict[sess_id]['extended']['cpoke_start'][fa_idxs_to_plot],
                     trial_event_dict[sess_id]['extended']['cpoke_out'][fa_idxs_to_plot],
                     kernel['bin_width'], title='FAs - extended')

    fig.suptitle('{} Single Units (n={:.0f})'.format(title_prefix, unit_info['filtered_single_units'].iloc[0]))

    # add legend entries
    line_colors = epoch_colors.values()
    point_shapes = [poke_start_shape, poke_end_shape]
    labels = ['Pre-poke', 'Poke', 'Post-poke', 'Poke Start', 'Poke End']

    lines = [Line2D([0], [0], color=c) for c in line_colors]
    lines.extend([Line2D([0], [0], marker=m, color='w', markerfacecolor='black') for m in point_shapes])

    fig.legend(lines, labels)
    plt.show()

    # plot all units
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)

    plot_pc_activity(axs[0,0], pca_dict[sess_id]['all']['poke'],
                     fr_dict[sess_id]['all']['poke_hits'][hit_trials_plot_sel],
                     trial_event_dict[sess_id]['poke']['trial_bounds'][hit_idxs_to_plot,0],
                     trial_event_dict[sess_id]['poke']['cpoke_start'][hit_idxs_to_plot],
                     trial_event_dict[sess_id]['poke']['cpoke_out'][hit_idxs_to_plot],
                     kernel['bin_width'], title='Hits - poke')

    plot_pc_activity(axs[1,0], pca_dict[sess_id]['all']['poke'],
                     fr_dict[sess_id]['all']['poke_fas'][fa_trials_plot_sel],
                     trial_event_dict[sess_id]['poke']['trial_bounds'][fa_idxs_to_plot,0],
                     trial_event_dict[sess_id]['poke']['cpoke_start'][fa_idxs_to_plot],
                     trial_event_dict[sess_id]['poke']['cpoke_out'][fa_idxs_to_plot],
                     kernel['bin_width'], title='FAs - poke')

    plot_pc_activity(axs[0,1], pca_dict[sess_id]['all']['extended'],
                     fr_dict[sess_id]['all']['extended_hits'][hit_trials_plot_sel],
                     trial_event_dict[sess_id]['extended']['trial_bounds'][hit_idxs_to_plot,0],
                     trial_event_dict[sess_id]['extended']['cpoke_start'][hit_idxs_to_plot],
                     trial_event_dict[sess_id]['extended']['cpoke_out'][hit_idxs_to_plot],
                     kernel['bin_width'], title='Hits - extended')

    plot_pc_activity(axs[1,1], pca_dict[sess_id]['all']['extended'],
                     fr_dict[sess_id]['all']['extended_fas'][fa_trials_plot_sel],
                     trial_event_dict[sess_id]['extended']['trial_bounds'][fa_idxs_to_plot,0],
                     trial_event_dict[sess_id]['extended']['cpoke_start'][fa_idxs_to_plot],
                     trial_event_dict[sess_id]['extended']['cpoke_out'][fa_idxs_to_plot],
                     kernel['bin_width'], title='FAs - extended')

    fig.suptitle('{} All Units (n={:.0f})'.format(title_prefix, unit_info['filtered_all_units'].iloc[0]))

    line_colors = epoch_colors.values()
    point_shapes = [poke_start_shape, poke_end_shape]
    labels = ['Pre-poke', 'Poke', 'Post-poke', 'Poke Start', 'Poke End']

    lines = [Line2D([0], [0], color=c) for c in line_colors]
    lines.extend([Line2D([0], [0], marker=m, color='w', markerfacecolor='black') for m in point_shapes])

    fig.legend(lines, labels)
    plt.show()

    # Replot the same two figures but now color by trial in session
    trial_colors = mpl.colormaps['viridis']

    # single units
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)

    plot_pc_activity(axs[0,0], pca_dict[sess_id]['single']['poke'],
                     fr_dict[sess_id]['single']['poke_hits'][hit_trials_plot_sel],
                     trial_event_dict[sess_id]['poke']['trial_bounds'][hit_idxs_to_plot,0],
                     trial_event_dict[sess_id]['poke']['cpoke_start'][hit_idxs_to_plot],
                     trial_event_dict[sess_id]['poke']['cpoke_out'][hit_idxs_to_plot],
                     kernel['bin_width'], title='Hits - poke',
                     color_type='trial', colors=trial_colors, norm_trial_idxs=hit_idxs_to_plot/n_trials)

    plot_pc_activity(axs[1,0], pca_dict[sess_id]['single']['poke'],
                     fr_dict[sess_id]['single']['poke_fas'][fa_trials_plot_sel],
                     trial_event_dict[sess_id]['poke']['trial_bounds'][fa_idxs_to_plot,0],
                     trial_event_dict[sess_id]['poke']['cpoke_start'][fa_idxs_to_plot],
                     trial_event_dict[sess_id]['poke']['cpoke_out'][fa_idxs_to_plot],
                     kernel['bin_width'], title='FAs - poke',
                     color_type='trial', colors=trial_colors, norm_trial_idxs=fa_idxs_to_plot/n_trials)

    plot_pc_activity(axs[0,1], pca_dict[sess_id]['single']['extended'],
                     fr_dict[sess_id]['single']['extended_hits'][hit_trials_plot_sel],
                     trial_event_dict[sess_id]['extended']['trial_bounds'][hit_idxs_to_plot,0],
                     trial_event_dict[sess_id]['extended']['cpoke_start'][hit_idxs_to_plot],
                     trial_event_dict[sess_id]['extended']['cpoke_out'][hit_idxs_to_plot],
                     kernel['bin_width'], title='Hits - extended',
                     color_type='trial', colors=trial_colors, norm_trial_idxs=hit_idxs_to_plot/n_trials)

    plot_pc_activity(axs[1,1], pca_dict[sess_id]['single']['extended'],
                     fr_dict[sess_id]['single']['extended_fas'][fa_trials_plot_sel],
                     trial_event_dict[sess_id]['extended']['trial_bounds'][fa_idxs_to_plot,0],
                     trial_event_dict[sess_id]['extended']['cpoke_start'][fa_idxs_to_plot],
                     trial_event_dict[sess_id]['extended']['cpoke_out'][fa_idxs_to_plot],
                     kernel['bin_width'], title='FAs - extended',
                     color_type='trial', colors=trial_colors, norm_trial_idxs=fa_idxs_to_plot/n_trials)

    fig.suptitle('{} Single Units (n={:.0f})'.format(title_prefix, unit_info['filtered_single_units'].iloc[0]))

    # add color bar
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=1, vmax=n_trials), cmap=trial_colors),
                 ax=axs.ravel().tolist(), label='Trial #')

    # add legend
    point_shapes = [poke_start_shape, poke_end_shape]
    labels = ['Poke Start', 'Poke End']
    lines = [Line2D([0], [0], marker=m, color='w', markerfacecolor='black') for m in point_shapes]
    fig.legend(lines, labels)
    plt.show()

    # all units
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)

    plot_pc_activity(axs[0,0], pca_dict[sess_id]['all']['poke'],
                     fr_dict[sess_id]['all']['poke_hits'][hit_trials_plot_sel],
                     trial_event_dict[sess_id]['poke']['trial_bounds'][hit_idxs_to_plot,0],
                     trial_event_dict[sess_id]['poke']['cpoke_start'][hit_idxs_to_plot],
                     trial_event_dict[sess_id]['poke']['cpoke_out'][hit_idxs_to_plot],
                     kernel['bin_width'], title='Hits - poke',
                     color_type='trial', colors=trial_colors, norm_trial_idxs=hit_idxs_to_plot/n_trials)

    plot_pc_activity(axs[1,0], pca_dict[sess_id]['all']['poke'],
                     fr_dict[sess_id]['all']['poke_fas'][fa_trials_plot_sel],
                     trial_event_dict[sess_id]['poke']['trial_bounds'][fa_idxs_to_plot,0],
                     trial_event_dict[sess_id]['poke']['cpoke_start'][fa_idxs_to_plot],
                     trial_event_dict[sess_id]['poke']['cpoke_out'][fa_idxs_to_plot],
                     kernel['bin_width'], title='FAs - poke',
                     color_type='trial', colors=trial_colors, norm_trial_idxs=fa_idxs_to_plot/n_trials)

    plot_pc_activity(axs[0,1], pca_dict[sess_id]['all']['extended'],
                     fr_dict[sess_id]['all']['extended_hits'][hit_trials_plot_sel],
                     trial_event_dict[sess_id]['extended']['trial_bounds'][hit_idxs_to_plot,0],
                     trial_event_dict[sess_id]['extended']['cpoke_start'][hit_idxs_to_plot],
                     trial_event_dict[sess_id]['extended']['cpoke_out'][hit_idxs_to_plot],
                     kernel['bin_width'], title='Hits - extended',
                     color_type='trial', colors=trial_colors, norm_trial_idxs=hit_idxs_to_plot/n_trials)

    plot_pc_activity(axs[1,1], pca_dict[sess_id]['all']['extended'],
                     fr_dict[sess_id]['all']['extended_fas'][fa_trials_plot_sel],
                     trial_event_dict[sess_id]['extended']['trial_bounds'][fa_idxs_to_plot,0],
                     trial_event_dict[sess_id]['extended']['cpoke_start'][fa_idxs_to_plot],
                     trial_event_dict[sess_id]['extended']['cpoke_out'][fa_idxs_to_plot],
                     kernel['bin_width'], title='FAs - extended',
                     color_type='trial', colors=trial_colors, norm_trial_idxs=fa_idxs_to_plot/n_trials)

    fig.suptitle('{} All Units (n={:.0f})'.format(title_prefix, unit_info['filtered_all_units'].iloc[0]))

    # add color bar
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=1, vmax=n_trials), cmap=trial_colors),
                 ax=axs.ravel().tolist(), label='Trial #')

    # add legend
    point_shapes = [poke_start_shape, poke_end_shape]
    labels = ['Poke Start', 'Poke End']
    lines = [Line2D([0], [0], marker=m, color='w', markerfacecolor='black') for m in point_shapes]
    fig.legend(lines, labels)
    plt.show()

# %% Plot PCA for each session including different time periods and different numbers of units

n_plot_trials = 25
start_idx = 0

sess_unit_info = unit_data.groupby(['subjid','sessid']).agg(
    single_units=('single_unit', 'sum'), all_units=('single_unit', 'count')
    ).sort_values(['subjid','single_units'], ascending=False).reset_index()
sess_ids = sess_unit_info['sessid'].to_numpy()

kernel = ephys_utils.get_filter_kernel(width=0.3, bin_width=0.05)  # defaults to causal half gaussian with a width of 0.2 and bin width of 5e-3

# calculate the extended trial bounds as some time before the poke and some time after
t_before_poke = 2
t_after_poke = 3

# all of these dictionaries are first indexed by session id, and then by the following fields
# first single/all, then poke_hits/poke_fas/extended_hits/extended_fas
fr_dict = {sess_id: {'single': {}, 'all':{}} for sess_id in sess_ids}
# first single/all, then poke/extended
pca_dict = {sess_id: {'single': {}, 'all':{}} for sess_id in sess_ids}
# hit_select/fa_select/hit_idxs/fa_idxs
trial_select_dict = {sess_id: {} for sess_id in sess_ids}
# trial_bounds/cpoke_start/cpoke_out
trial_event_dict = {sess_id: {'poke': {}, 'extended': {}} for sess_id in sess_ids}

for i, sess_id in enumerate(sess_ids[start_idx:]):
    start = time.perf_counter()

    sess_trials = sess_data[sess_data['sessid'] == sess_id].reset_index()
    n_trials = len(sess_trials)

    # to save memory, ignore units that have less spikes than the number of trials
    all_units = unit_data[(unit_data['sessid'] == sess_id) & (unit_data['number_spikes'] > n_trials)].reset_index(drop=True)
    single_units = all_units[all_units['single_unit'] == 1].reset_index(drop=True)
    sess_unit_info.loc[sess_unit_info['sessid'] == sess_id, 'filtered_single_units'] = len(single_units)
    sess_unit_info.loc[sess_unit_info['sessid'] == sess_id, 'filtered_all_units'] = len(all_units)

    # compute the trial selects and indices
    hit_select = sess_trials['hit'] == 1
    # ignore FAs that are too fast to be driven by the stimulus
    fa_select = (sess_trials['FA'] == 1) & (sess_trials['cpoke_dur'] > 0.5)
    trial_idxs = np.arange(n_trials)
    trial_select_dict[sess_id]['hit_select'] = hit_select
    trial_select_dict[sess_id]['fa_select'] = fa_select
    trial_select_dict[sess_id]['hit_idxs'] = trial_idxs[hit_select]
    trial_select_dict[sess_id]['fa_idxs'] = trial_idxs[fa_select]

    # get some basic timestamp info
    trial_start_ts = single_units['trial_start_timestamps'].iloc[0][0:-1]
    # note: last trial start is given by the ttl pulse at the start of the trial that is incomplete at the end of the session so ignore it
    single_spike_ts = single_units['spike_timestamps']
    all_spike_ts = all_units['spike_timestamps']

    cpoke_start = sess_trials['cpoke_start'].to_numpy()
    cpoke_out = sess_trials['cpoke_out'].to_numpy()
    poke_bounds = sess_trials[['cpoke_start', 'cpoke_out']].to_numpy()

    # compute the cpoke events in absolute time from start of session
    abs_cpoke_start = trial_start_ts + cpoke_start
    abs_cpoke_out = trial_start_ts + cpoke_out

    abs_prepoke_start = abs_cpoke_start - t_before_poke # this will be an effective re-zeroing of the trial start for trial spike assignment
    abs_postpoke_end = abs_cpoke_out + t_after_poke

    # for trials with periods between trials shorter than the combined window durations,
    # choose a midpoint based on relative durations of pre and post poke windows
    interpoke_dur = abs_cpoke_start[1:] - abs_cpoke_out[:-1]
    short_interpoke_sel = interpoke_dur < (t_after_poke + t_before_poke)
    short_interpoke_sel_pre = np.concatenate([[False], short_interpoke_sel])
    short_interpoke_sel_post = np.concatenate([short_interpoke_sel, [False]])

    abs_prepoke_start[short_interpoke_sel_pre] = abs_cpoke_start[short_interpoke_sel_pre] - interpoke_dur[short_interpoke_sel]*t_before_poke/(t_after_poke + t_before_poke)
    abs_postpoke_end[short_interpoke_sel_post] = abs_prepoke_start[short_interpoke_sel_pre]

    # compute reference points for these re-zeroed trials
    trial_cpoke_start = abs_cpoke_start - abs_prepoke_start
    trial_cpoke_out = abs_cpoke_out - abs_prepoke_start
    trial_end = abs_postpoke_end - abs_prepoke_start
    trial_bounds = np.concatenate((np.zeros_like(trial_end.reshape(-1,1)), trial_end.reshape(-1,1)), axis=1)

    # persist trial event information
    trial_event_dict[sess_id]['poke']['cpoke_start'] = cpoke_start
    trial_event_dict[sess_id]['poke']['cpoke_out'] = cpoke_out
    trial_event_dict[sess_id]['poke']['trial_bounds'] = poke_bounds

    trial_event_dict[sess_id]['extended']['cpoke_start'] = trial_cpoke_start
    trial_event_dict[sess_id]['extended']['cpoke_out'] = trial_cpoke_out
    trial_event_dict[sess_id]['extended']['trial_bounds'] = trial_bounds

    # compute the smmothed firing rate matrices
    fr_dict[sess_id]['single']['poke_hits'] = ephys_utils.get_fr_matrix_by_trial(single_spike_ts, trial_start_ts, kernel, poke_bounds, hit_select)

    fr_dict[sess_id]['single']['poke_fas'] = ephys_utils.get_fr_matrix_by_trial(single_spike_ts, trial_start_ts, kernel, poke_bounds, fa_select)

    fr_dict[sess_id]['single']['extended_hits'] = ephys_utils.get_fr_matrix_by_trial(single_spike_ts, abs_prepoke_start, kernel, trial_bounds, hit_select)

    fr_dict[sess_id]['single']['extended_fas'] = ephys_utils.get_fr_matrix_by_trial(single_spike_ts, abs_prepoke_start, kernel, trial_bounds, fa_select)

    fr_dict[sess_id]['all']['poke_hits'] = ephys_utils.get_fr_matrix_by_trial(all_spike_ts, trial_start_ts, kernel, poke_bounds, hit_select)

    fr_dict[sess_id]['all']['poke_fas'] = ephys_utils.get_fr_matrix_by_trial(all_spike_ts, trial_start_ts, kernel, poke_bounds, fa_select)

    fr_dict[sess_id]['all']['extended_hits'] = ephys_utils.get_fr_matrix_by_trial(all_spike_ts, abs_prepoke_start, kernel, trial_bounds, hit_select)

    fr_dict[sess_id]['all']['extended_fas'] = ephys_utils.get_fr_matrix_by_trial(all_spike_ts, abs_prepoke_start, kernel, trial_bounds, fa_select)

    # perform PCA on the smoothed firing rate matrices
    pca_dict[sess_id]['single']['poke'] = PCA()
    pca_dict[sess_id]['single']['poke'].fit(np.vstack((np.vstack(fr_dict[sess_id]['single']['poke_hits']),
                                                        np.vstack(fr_dict[sess_id]['single']['poke_fas']))))

    pca_dict[sess_id]['single']['extended'] = PCA()
    pca_dict[sess_id]['single']['extended'].fit(np.vstack((np.vstack(fr_dict[sess_id]['single']['extended_hits']),
                                                            np.vstack(fr_dict[sess_id]['single']['extended_fas']))))

    pca_dict[sess_id]['all']['poke'] = PCA()
    pca_dict[sess_id]['all']['poke'].fit(np.vstack((np.vstack(fr_dict[sess_id]['all']['poke_hits']),
                                                    np.vstack(fr_dict[sess_id]['all']['poke_fas']))))

    pca_dict[sess_id]['all']['extended'] = PCA()
    pca_dict[sess_id]['all']['extended'].fit(np.vstack((np.vstack(fr_dict[sess_id]['all']['extended_hits']),
                                                        np.vstack(fr_dict[sess_id]['all']['extended_fas']))))

    print('Completed session {0} in {3:.1f} s. {1}/{2} sessions.'.format(sess_id, i+1, len(sess_ids), time.perf_counter()-start))

    compare_pc_act(sess_id, sess_unit_info, trial_select_dict, pca_dict, fr_dict, trial_event_dict, kernel, n_plot_trials)

# %% Single Prints

sess_id = 49315

compare_pc_act(sess_id, sess_unit_info, trial_select_dict, pca_dict, fr_dict, trial_event_dict, kernel, n_plot_trials)