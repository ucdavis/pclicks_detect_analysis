# -*- coding: utf-8 -*-
"""
Example of how to do PCA

@author: tanner stevenson
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

rng = default_rng(1)

# %% Load the data

loc_db = db.LocalDB_PClicksDetect()
# subj_ids = loc_db.get_protocol_subject_ids() # will get all subjects recorded during this protocol
# for now lets just use these ones
subj_ids = [78]  # , 94

# get all unit and session data for these subjects
# this will take a while (~60 mins) the first time
unit_data = loc_db.get_subj_unit_data(subj_ids)
sess_data = loc_db.get_subj_behavior_data(subj_ids)

# %% Filter unit data to one session

# lets get filtered firing rates for all units in a session while the rat is poked into the center port on hit and miss trials
# find the session with the most single units
sess_single_units = unit_data.groupby('sessid')['single_unit'].sum()
max_units_sess_id = sess_single_units.index[np.argmax(sess_single_units)]

# trim down the unit and session data
sess_units = unit_data[(unit_data['sessid'] == max_units_sess_id) & (unit_data['single_unit'] == 1)].reset_index()
sess_trials = sess_data[sess_data['sessid'] == max_units_sess_id].reset_index()
n_trials = len(sess_trials)

# %% Construct smoothed firing rate matrices (units X time) for each trial in the session

# we will construct smoothed firing rates separately for just the poke duration and for some time pre and post poke to include some task-specific movement and reward

kernel = ephys_utils.get_filter_kernel()  # defaults to causal half gaussian with a width of 0.2 and bin width of 5e-3
hit_select = sess_trials['hit'] == 1
# ignore FAs that are too fast to be driven by the stimulus
fa_select = (sess_trials['FA'] == 1) & (sess_trials['cpoke_dur'] > 0.5)

# get some basic timestamp info
trial_start_ts = sess_units['trial_start_timestamps'].iloc[0][0:-1]
# note: last trial start is given by the ttl pulse at the start of the trial that is incomplete at the end of the session so ignore it
spike_ts = sess_units['spike_timestamps']
cpoke_start = sess_trials['cpoke_start'].to_numpy()
cpoke_out = sess_trials['cpoke_out'].to_numpy()
poke_bounds = sess_trials[['cpoke_start', 'cpoke_out']]

# calculate the perceived trial bounds as some time before the poke and some time after
t_before_poke = 2
t_after_poke = 3
# compute the cpoke events in absolute time from start of session
abs_cpoke_start = trial_start_ts + cpoke_start
abs_cpoke_out = trial_start_ts + cpoke_out

abs_prepoke_start = abs_cpoke_start - t_before_poke # this will be an effective re-zeroing of the trial start for trial spike assignment
abs_postpoke_end = abs_cpoke_out + t_after_poke

# for trials with periods between trials shorter than the combined window durations,
# choose a midpoint based on relative durations of pre and post poke windows
interpoke_dur = abs_cpoke_start[1:] - abs_cpoke_out[:-1]
short_interpoke_sel = interpoke_dur < t_after_poke + t_before_poke
short_interpoke_sel_pre = np.concatenate([[False], short_interpoke_sel])
short_interpoke_sel_post = np.concatenate([short_interpoke_sel, [False]])

abs_prepoke_start[short_interpoke_sel_pre] = abs_cpoke_start[short_interpoke_sel_pre] - interpoke_dur[short_interpoke_sel]*t_before_poke/(t_after_poke + t_before_poke)
abs_postpoke_end[short_interpoke_sel_post] = abs_prepoke_start[short_interpoke_sel_pre]

# compute reference points for these re-zeroed trials
trial_cpoke_start = abs_cpoke_start - abs_prepoke_start
trial_cpoke_out = abs_cpoke_out - abs_prepoke_start
trial_end = abs_postpoke_end - abs_prepoke_start
trial_bounds = np.concatenate((np.zeros_like(trial_end.reshape(-1,1)), trial_end.reshape(-1,1)), axis=1)

# compute the relative time from start of previous trial to the nosepoke of the next trial

smoothed_frs_hits_poke = ephys_utils.get_fr_matrix_by_trial(spike_ts, trial_start_ts, kernel, poke_bounds, hit_select)

smoothed_frs_fas_poke = ephys_utils.get_fr_matrix_by_trial(spike_ts, trial_start_ts, kernel, poke_bounds, fa_select)

smoothed_frs_hits_trial = ephys_utils.get_fr_matrix_by_trial(spike_ts, abs_prepoke_start, kernel, trial_bounds, hit_select)

smoothed_frs_fas_trial = ephys_utils.get_fr_matrix_by_trial(spike_ts, abs_prepoke_start, kernel, trial_bounds, fa_select)


# %% Perform PCA

# stack all smoothed firing matrices into one large fr matrix with units in the columns
joined_unit_frs_poke = np.vstack((np.vstack(smoothed_frs_hits_poke), np.vstack(smoothed_frs_fas_poke)))
joined_unit_frs_trial = np.vstack((np.vstack(smoothed_frs_hits_trial), np.vstack(smoothed_frs_fas_trial)))

# perform PCA
pca_poke = PCA()
pca_poke.fit(joined_unit_frs_poke)

pca_trial = PCA()
pca_trial.fit(joined_unit_frs_trial)

# %% Declare reusable plotting method
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


# %% plot activity from subset of trials in PC space

# select random individual trials to plot in PC space
n_plot_trials = 10
trial_idxs = np.arange(n_trials)
hit_idxs = trial_idxs[hit_select]
fa_idxs = trial_idxs[fa_select]

hit_idxs_to_plot = np.sort(rng.choice(hit_idxs, n_plot_trials, replace=False))
fa_idxs_to_plot = np.sort(rng.choice(fa_idxs, n_plot_trials, replace=False))

hit_trials_plot_sel = [idx in hit_idxs_to_plot for idx in hit_idxs]
fa_trials_plot_sel = [idx in fa_idxs_to_plot for idx in fa_idxs]

# # compute time-dilated trial average to plot in PC space
# # first find the number of bins for each epoch based on the number of minimum bin widths within the minimum duration of each epoch
# min_bin_width = 0.1
# cpoke_bins_hit = np.floor(np.min(sess_trials.loc[hit_select, 'cpoke_dur'])/min_bin_width)
# cpoke_bins_fa = np.floor(np.min(sess_trials.loc[fa_select, 'cpoke_dur'])/min_bin_width)
# prepoke_bins_hit = np.floor(np.min(trial_cpoke_start[hit_select])/min_bin_width)
# prepoke_bins_fa = np.floor(np.min(trial_cpoke_start[fa_select])/min_bin_width)
# postpoke_bins_hit = np.floor(np.min(trial_end[hit_select] - trial_cpoke_out[hit_select])/min_bin_width)
# postpoke_bins_fa = np.floor(np.min(trial_end[fa_select] - trial_cpoke_out[fa_select])/min_bin_width)


# Create 2x2 figure - hits & fas in different rows, poke and trial in different columns
fig, axs = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)

plot_pc_activity(axs[0,0], pca_poke, smoothed_frs_hits_poke[hit_trials_plot_sel], cpoke_start[hit_idxs_to_plot],
                 cpoke_start[hit_idxs_to_plot], cpoke_out[hit_idxs_to_plot],
                 kernel['bin_width'], title='Hits - center poke')
plot_pc_activity(axs[1,0], pca_poke, smoothed_frs_fas_poke[fa_trials_plot_sel], cpoke_start[fa_idxs_to_plot],
                 cpoke_start[fa_idxs_to_plot], cpoke_out[fa_idxs_to_plot],
                 kernel['bin_width'], title='FAs - center poke')

plot_pc_activity(axs[0,1], pca_trial, smoothed_frs_hits_trial[hit_trials_plot_sel], trial_bounds[hit_idxs_to_plot,0],
                 trial_cpoke_start[hit_idxs_to_plot], trial_cpoke_out[hit_idxs_to_plot],
                 kernel['bin_width'], title='Hits - expanded trial')
plot_pc_activity(axs[1,1], pca_trial, smoothed_frs_fas_trial[fa_trials_plot_sel], trial_bounds[fa_idxs_to_plot,0],
                 trial_cpoke_start[fa_idxs_to_plot], trial_cpoke_out[fa_idxs_to_plot],
                 kernel['bin_width'], title='FAs - expanded trial')

# add legend entries
line_colors = epoch_colors.values()
point_shapes = [poke_start_shape, poke_end_shape]
labels = ['Pre-poke', 'Poke', 'Post-poke', 'Poke Start', 'Poke End']

lines = [Line2D([0], [0], color=c) for c in line_colors]
lines.extend([Line2D([0], [0], marker=m, color='w', markerfacecolor='black') for m in point_shapes])

fig.legend(lines, labels)
plt.show()

# %% Replot but now color by trial in session
trial_colors = mpl.colormaps['viridis']

fig, axs = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)


plot_pc_activity(axs[0,0], pca_poke, smoothed_frs_hits_poke[hit_trials_plot_sel], cpoke_start[hit_idxs_to_plot],
                 cpoke_start[hit_idxs_to_plot], cpoke_out[hit_idxs_to_plot], kernel['bin_width'],
                 title='Hits - center poke', color_type='trial', colors=trial_colors, norm_trial_idxs=hit_idxs_to_plot/n_trials)

plot_pc_activity(axs[1,0], pca_poke, smoothed_frs_fas_poke[fa_trials_plot_sel], cpoke_start[fa_idxs_to_plot],
                 cpoke_start[fa_idxs_to_plot], cpoke_out[fa_idxs_to_plot], kernel['bin_width'],
                 title='FAs - center poke', color_type='trial', colors=trial_colors, norm_trial_idxs=fa_idxs_to_plot/n_trials)

plot_pc_activity(axs[0,1], pca_trial, smoothed_frs_hits_trial[hit_trials_plot_sel], trial_bounds[hit_idxs_to_plot,0],
                 trial_cpoke_start[hit_idxs_to_plot], trial_cpoke_out[hit_idxs_to_plot], kernel['bin_width'],
                 title='Hits - expanded trial', color_type='trial', colors=trial_colors, norm_trial_idxs=hit_idxs_to_plot/n_trials)

plot_pc_activity(axs[1,1], pca_trial, smoothed_frs_fas_trial[fa_trials_plot_sel], trial_bounds[fa_idxs_to_plot,0],
                 trial_cpoke_start[fa_idxs_to_plot], trial_cpoke_out[fa_idxs_to_plot], kernel['bin_width'],
                 title='FAs - expanded trial', color_type='trial', colors=trial_colors, norm_trial_idxs=fa_idxs_to_plot/n_trials)

# add color bar
fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=1, vmax=n_trials), cmap=trial_colors),
             ax=axs.ravel().tolist(), label='Trial #')

# add legend
point_shapes = [poke_start_shape, poke_end_shape]
labels = ['Poke Start', 'Poke End']
lines = [Line2D([0], [0], marker=m, color='w', markerfacecolor='black') for m in point_shapes]
fig.legend(lines, labels)
plt.show()

# %% replot in the same way but using different PCs

# colored by trial epoch
fig, axs = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)

plot_pc_activity(axs[0,0], pca_poke, smoothed_frs_hits_poke[hit_trials_plot_sel], cpoke_start[hit_idxs_to_plot],
                 cpoke_start[hit_idxs_to_plot], cpoke_out[hit_idxs_to_plot],
                 kernel['bin_width'], title='Hits - center poke', x_pc=2, y_pc=3)
plot_pc_activity(axs[1,0], pca_poke, smoothed_frs_fas_poke[fa_trials_plot_sel], cpoke_start[fa_idxs_to_plot],
                 cpoke_start[fa_idxs_to_plot], cpoke_out[fa_idxs_to_plot],
                 kernel['bin_width'], title='FAs - center poke', x_pc=2, y_pc=3)

plot_pc_activity(axs[0,1], pca_trial, smoothed_frs_hits_trial[hit_trials_plot_sel], trial_bounds[hit_idxs_to_plot,0],
                 trial_cpoke_start[hit_idxs_to_plot], trial_cpoke_out[hit_idxs_to_plot],
                 kernel['bin_width'], title='Hits - expanded trial', x_pc=1, y_pc=3)
plot_pc_activity(axs[1,1], pca_trial, smoothed_frs_fas_trial[fa_trials_plot_sel], trial_bounds[fa_idxs_to_plot,0],
                 trial_cpoke_start[fa_idxs_to_plot], trial_cpoke_out[fa_idxs_to_plot],
                 kernel['bin_width'], title='FAs - expanded trial', x_pc=1, y_pc=3)

# add legend entries
line_colors = epoch_colors.values()
point_shapes = [poke_start_shape, poke_end_shape]
labels = ['Pre-poke', 'Poke', 'Post-poke', 'Poke Start', 'Poke End']

lines = [Line2D([0], [0], color=c) for c in line_colors]
lines.extend([Line2D([0], [0], marker=m, color='w', markerfacecolor='black') for m in point_shapes])

fig.legend(lines, labels)
plt.show()

# colored by trial
fig, axs = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)

plot_pc_activity(axs[0,0], pca_poke, smoothed_frs_hits_poke[hit_trials_plot_sel], cpoke_start[hit_idxs_to_plot],
                 cpoke_start[hit_idxs_to_plot], cpoke_out[hit_idxs_to_plot], kernel['bin_width'], title='Hits - center poke',
                 color_type='trial', colors=trial_colors, norm_trial_idxs=hit_idxs_to_plot/n_trials, x_pc=2, y_pc=3)

plot_pc_activity(axs[1,0], pca_poke, smoothed_frs_fas_poke[fa_trials_plot_sel], cpoke_start[fa_idxs_to_plot],
                 cpoke_start[fa_idxs_to_plot], cpoke_out[fa_idxs_to_plot], kernel['bin_width'], title='FAs - center poke',
                 color_type='trial', colors=trial_colors, norm_trial_idxs=fa_idxs_to_plot/n_trials, x_pc=2, y_pc=3)

plot_pc_activity(axs[0,1], pca_trial, smoothed_frs_hits_trial[hit_trials_plot_sel], trial_bounds[hit_idxs_to_plot,0],
                 trial_cpoke_start[hit_idxs_to_plot], trial_cpoke_out[hit_idxs_to_plot], kernel['bin_width'], title='Hits - expanded trial',
                 color_type='trial', colors=trial_colors, norm_trial_idxs=hit_idxs_to_plot/n_trials, x_pc=1, y_pc=3)

plot_pc_activity(axs[1,1], pca_trial, smoothed_frs_fas_trial[fa_trials_plot_sel], trial_bounds[fa_idxs_to_plot,0],
                 trial_cpoke_start[fa_idxs_to_plot], trial_cpoke_out[fa_idxs_to_plot], kernel['bin_width'], title='FAs - expanded trial',
                 color_type='trial', colors=trial_colors, norm_trial_idxs=fa_idxs_to_plot/n_trials, x_pc=1, y_pc=3)

# add color bar
fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=1, vmax=n_trials), cmap=trial_colors),
             ax=axs.ravel().tolist(), label='Trial #')

# add legend
point_shapes = [poke_start_shape, poke_end_shape]
labels = ['Poke Start', 'Poke End']
lines = [Line2D([0], [0], marker=m, color='w', markerfacecolor='black') for m in point_shapes]
fig.legend(lines, labels)
plt.show()
