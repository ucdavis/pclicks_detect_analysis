# -*- coding: utf-8 -*-
"""
Example of how to do PCA

@author: tanner stevenson
"""

import db_access
import local_db.pclicksdetect_db as db
import utils
import ephys_utils
import numpy as np
import pandas as pd

loc_db = db.LocalDB_PClicksDetect()
# subj_ids = loc_db.get_protocol_subject_ids() # will get all subjects recorded during this protocol
# for now lets just use these ones
subj_ids = [78]  # , 94

# get all unit and session data for these subjects
# this will take a while (~60 mins) the first time
unit_data = loc_db.get_subj_unit_data(subj_ids)
sess_data = loc_db.get_subj_behavior_data(subj_ids)

# lets get filtered firing rates for all units in a session while the rat is poked into the center port on hit and miss trials
# find the session with the most single units
sess_single_units = unit_data.groupby('sessid')['single_unit'].sum()
max_units_sess_id = sess_single_units.index[np.argmax(sess_single_units)]

# trim down the unit and session data
sess_units = unit_data[(unit_data['sessid'] == max_units_sess_id) & (unit_data['single_unit'] == 1)].reset_index()
sess_trials = sess_data[sess_data['sessid'] == max_units_sess_id].reset_index()

# One way to do it more manually and only smoothing trials of interest:

# # First get trial spike times for each unit
# all_trial_spikes = [ephys_utils.get_trial_spike_times(
#     sess_units['spike_timestamps'].iloc[i], sess_units['trial_start_timestamps'].iloc[i]) for i in range(len(sess_units))]

# # next we'll select for trials that resulted in hit or a miss
# hit_select = sess_trials['hit'] == 1
# miss_select = sess_trials['miss'] == 1

# # now we can get the smoothed firing rates for all units during either a hit or a miss
# kernel = ephys_utils.get_filter_kernel()  # defaults to causal half gaussian with a width of 0.2 and bin width of 5e-3
# smoothed_frs_hits = [[ephys_utils.get_smoothed_firing_rate(trial_spikes, kernel, sess_trials['cpoke_start'].iloc[i], sess_trials['cpoke_out'].iloc[i])[0]
#                       for i, trial_spikes in enumerate(unit_spikes) if hit_select[i]]
#                      for unit_spikes in all_trial_spikes]

# smoothed_frs_misses = [[ephys_utils.get_smoothed_firing_rate(trial_spikes, kernel, sess_trials['cpoke_start'].iloc[i], sess_trials['cpoke_out'].iloc[i])[0]
#                         for i, trial_spikes in enumerate(unit_spikes) if miss_select[i]]
#                        for unit_spikes in all_trial_spikes]

# # finally we can concatenate trials together for each unit, with units in the columns and concatenated signal timesteps in the rows
# joined_unit_frs = np.vstack((
#     np.concatenate([np.concatenate(unit_frs).reshape(-1, 1) for unit_frs in smoothed_frs_hits], axis=1),
#     np.concatenate([np.concatenate(unit_frs).reshape(-1, 1) for unit_frs in smoothed_frs_misses], axis=1)))

# Other way to do it with a convenient helper method:

# first get smoothed frs for all units over hits and misses
kernel = ephys_utils.get_filter_kernel()  # defaults to causal half gaussian with a width of 0.2 and bin width of 5e-3
hit_select = sess_trials['hit'] == 1
miss_select = sess_trials['miss'] == 1

smoothed_frs_hits = ephys_utils.get_fr_matrix_by_trial(
    sess_units, kernel, sess_trials[['cpoke_start', 'cpoke_out']], hit_select)

smoothed_frs_misses = ephys_utils.get_fr_matrix_by_trial(
    sess_units, kernel, sess_trials[['cpoke_start', 'cpoke_out']], miss_select)

# then stack all matrices into one large fr matrix with units in the columns
joined_unit_frs = np.vstack((np.vstack(smoothed_frs_hits), np.vstack(smoothed_frs_misses)))
