# -*- coding: utf-8 -*-
"""
Example of how to do calculate and plot various PSTHs

@author: tanner stevenson
"""

# %% Imports

import sys
from os import path
sys.path.append(path.join(path.dirname(path.abspath(__file__)), '..'))

import numpy as np
import matplotlib.pyplot as plt
import hankslab_db.pclicksdetect_db as db
from sys_neuro_tools import ephys_utils, plot_utils

# %% Load the data

loc_db = db.LocalDB_PClicksDetect()
# subj_ids = loc_db.get_protocol_subject_ids() # will get all subjects recorded during this protocol
# for now lets just use these ones
subj_ids = [78]  # 94

# get all unit and session data for these subjects
# this will take a while (~60 mins) the first time
unit_data = loc_db.get_subj_unit_data(subj_ids)
sess_data = loc_db.get_subj_behavior_data(subj_ids)

# %% Filter the data

# lets get filtered firing rates for all units in a session while the rat is poked into the center port on hit and miss trials
# find the session with the most single units
sess_single_units = unit_data.groupby('sessid')['single_unit'].sum()
max_units_sess_id = sess_single_units.index[np.argmax(sess_single_units)]

# trim down the unit and session data
sess_units = unit_data[(unit_data['sessid'] == max_units_sess_id) & (unit_data['single_unit'] == 1)].reset_index()
sess_trials = sess_data[sess_data['sessid'] == max_units_sess_id].reset_index()

# find the unit with the most spikes just for demo purposes
max_spikes_unit = sess_units.iloc[np.argmax(sess_units['number_spikes'])]

# get trial spike times for each unit
trial_spikes = ephys_utils.get_trial_spike_times(
    max_spikes_unit['spike_timestamps'], max_spikes_unit['trial_start_timestamps'][0:-1])

# %% Create some PSTHs aligned to different points

# next we'll select for trials that resulted in hit or a fa
hit_select = (sess_trials['hit'] == 1) & (sess_trials['rewarded'])
fa_select = sess_trials['FA'] == 1

# now lets generate and plot PSTHs aligned to change and response times for both types of trials
kernel = ephys_utils.get_filter_kernel()  # defaults to causal half gaussian with a width of 0.2s and a bin width of 5e-3
hit_change_psth = ephys_utils.get_psth(trial_spikes[hit_select],
                                       sess_trials.loc[hit_select, 'change_time'],  # align to change time
                                       (-1, 0.5),  # show 1 second window around the change time
                                       kernel,
                                       sess_trials.loc[hit_select, ['stim_start', 'stim_end']])  # mask any signal outside the stimulus presentation

hit_resp_psth = ephys_utils.get_psth(trial_spikes[hit_select],
                                     sess_trials.loc[hit_select, 'cpoke_out'],  # align to response time
                                     (-1, 0.5),  # show 1 second before and 0.5 seconds after the response time
                                     kernel,
                                     sess_trials.loc[hit_select, ['stim_start', 'reward_time']])  # mask any signal before the stimulus and after the reward

fa_resp_psth = ephys_utils.get_psth(trial_spikes[fa_select],
                                    sess_trials.loc[fa_select, 'cpoke_out'],  # align to response time
                                    (-1, 0.5),  # show 1 second before and 0.5 seconds after the response time
                                    kernel,
                                    sess_trials.loc[fa_select, ['stim_start', 'trial_end']])  # mask any signal before the stimulus and after the reward

# %% Plot PSTHs
_, axs = plt.subplots(2, 2, constrained_layout=True)
axs = axs.flatten()
plot_utils.plot_psth_dict(hit_change_psth, axs[0])
plot_utils.plot_psth_dict(hit_resp_psth, axs[1])
plot_utils.plot_psth_dict(fa_resp_psth, axs[3])

axs[0].set_title('Hits')
axs[0].set_xlabel('Time from change (s)')
axs[0].set_ylabel('Firing rate (Hz)')
axs[1].set_title('Hits')
axs[1].set_xlabel('Time from response (s)')
axs[1].set_ylabel('Firing rate (Hz)')
axs[3].set_title('FAs')
axs[3].set_xlabel('Time from response (s)')
axs[3].set_ylabel('Firing rate (Hz)')

plt.show()

# We can also plot a raster of the psth
_, axs = plt.subplots(2, 1, sharex=True, constrained_layout=True)
axs = axs.flatten()
plot_utils.plot_psth_dict(hit_resp_psth, axs[0])
plot_utils.plot_raster(hit_resp_psth['aligned_spikes'], axs[1])

axs[0].set_title('Hits')
axs[0].set_ylabel('Firing rate (Hz)')
axs[1].set_xlabel('Time from response (s)')
axs[1].set_ylabel('Trial')

# as a sanity check, we can also make sure the spikes on individual trials line up with the smoothed signal from that trial
for i in range(4):
    _, axs = plt.subplots(2, 1, sharex=True, constrained_layout=True)
    plot_utils.plot_psth(hit_resp_psth['all_signals'][i], hit_resp_psth['time'], ax=axs[0])
    plot_utils.plot_raster(hit_resp_psth['aligned_spikes'][i], axs[1])

    axs[0].set_ylabel('Firing rate (Hz)')
    axs[1].set_ylabel('Trial')
    axs[1].set_xlabel('Time from response (s)')

plt.show()

# %% Create Click-triggered Average (CTA)

# we can also plot a click-triggered average which is the average response of the neuron to a click
# first calculate the boundaries over which we want to calculate the CTA for each trial
buffer = 0.1
# only include clicks a few ms after the stimulus starts and a few ms before the change when a change occurs
# or a few ms before the end of the stimulus otherwise
cta_start = sess_trials['stim_start'] + buffer
cta_end = sess_trials.apply(lambda x: x['change_time'] - buffer
                            if not np.isnan(x['change_time'])
                            else x['stim_end'] - buffer,
                            axis=1)
kernel = ephys_utils.get_filter_kernel()  # defaults to causal half gaussian with a width of 0.2s and a bin width of 5e-3
cta = ephys_utils.get_psth(trial_spikes,  # we'll use all trials
                           sess_trials['abs_click_times'],  # align to all clicks
                           (-0.4, 0.8),  # show 0.4 seconds before and 0.8 seconds after a click
                           kernel, [cta_start, cta_end])

# %% Plot Click-triggered Average (CTA)

_, axs = plt.subplots(2, 1, sharex=True, constrained_layout=True)
plot_utils.plot_psth_dict(cta, axs[0])
plot_utils.plot_raster(cta['aligned_spikes'][0:1000], axs[1]) # only plot 1000 clicks for time sake
axs[0].set_title('Click Triggered Average')
axs[0].set_ylabel('Firing rate (Hz)')
axs[1].set_xlabel('Time from click (s)')
axs[1].set_ylabel('Trial')

plt.show()

# %%
