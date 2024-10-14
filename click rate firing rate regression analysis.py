# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 14:54:21 2024

@author: tanne
"""

# %% imports

import init
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from pyutils import utils
from hankslab_db import db_access
import hankslab_db.pclicksdetect_db as db
from sys_neuro_tools import ephys_utils, plot_utils, fp_utils
import scipy.signal as sig_tools
from scipy import optimize
from sklearn import linear_model
import sklearn.model_selection as skl_ms
from sklearn.metrics import root_mean_squared_error, r2_score, make_scorer
import random as rand
import copy as cp
import time
import os.path as path
import pickle
import warnings


# %% Load the data

loc_db = db.LocalDB_PClicksDetect()
# subj_ids = loc_db.get_protocol_subject_ids('PClicksDetect') # will get all subjects recorded during this protocol
# for now lets just use these ones
subj_ids = [78, 94, 142, 136, 150, 145]  # 78, 94

sess_ids = db_access.get_subj_unit_sess_ids(subj_ids)
sess_ids = [47622]

# get all session data for these subjects
unit_data = loc_db.get_sess_unit_data(utils.flatten(sess_ids))
sess_data = loc_db.get_behavior_data(utils.flatten(sess_ids))

# %% define reusable methods

def get_base_cr(df):
    pre_change_cpoke_time = df.apply(lambda x: np.min([x['cpoke_out']-x['stim_start'], x['change_delay']]), axis=1)
    n_clicks = [np.sum(df['rel_click_times'].iloc[i] < t) for i, t in enumerate(pre_change_cpoke_time)]
    return np.sum(n_clicks)/np.sum(pre_change_cpoke_time)

def find_peak_start(signal, t):
    peaks, _ = sig_tools.find_peaks(signal)

    if len(peaks) > 0:
        peak_idx = peaks[np.argmax(signal[peaks])]
    else:
        peak_idx = np.argmax(signal)

    # find inflection point using a peicewise linear fit
    # x0, y0 is the x/y of the junction which will be considered the inflection
    def piecewise_linear(x, x0, y0, k1, k2):
        return np.piecewise(x, [x < x0], [lambda x:k1*x + y0 - k1*x0, lambda x:k2*x + y0 - k2*x0])

    bounds = ([        t[0], np.min(signal), -np.inf, -np.inf],
              [ t[peak_idx], np.max(signal),  np.inf, np.inf])

    p = optimize.curve_fit(piecewise_linear, t[:peak_idx+1], signal[:peak_idx+1], bounds=bounds)[0]
    inflx_idx = np.argmin(np.abs(t - p[0]))

    # xd = np.linspace(t[0], t[peak_idx], 100)
    # _, ax = plt.subplots(1,1)
    # ax.plot(t, signal)
    # ax.plot(t[peak_idx], signal[peak_idx], marker=7, markersize=10, color='C1')
    # ax.plot(xd, piecewise_linear(xd, *p), color='black', linestyle='dashed')
    # ax.plot(t[inflx_idx], signal[inflx_idx], marker=6, markersize=10, color='C2')

    return inflx_idx
    
# %% Investigate response click reverse correlations by session

bin_width = 0.02
rc_window = np.array([-0.8, 0])
kernel = ephys_utils.get_filter_kernel(filter_type='none', bin_width=bin_width)
filt_f = 5

trial_sel = 'fa' # 'fa' 'hit' 'all'

for sess_id in [47622, 67430]: #utils.flatten(sess_ids):
    sess_trials = sess_data[sess_data['sessid'] == sess_id]
    
    fa_sel = ((sess_trials['FA'] == 1) & (sess_trials['cpoke_dur'] > 0.5)) & sess_trials['valid']
    hit_sel = (sess_trials['hit'] == 1) & sess_trials['valid']
    all_sel = fa_sel | hit_sel 
    
    # compute average pre-change firing rate for response trials as baseline to subtract from RC kernels
    avg_cr = get_base_cr(sess_trials[all_sel])
    
    match trial_sel:
        case 'fa':
            resp_sel = fa_sel
        case 'hit':
            resp_sel = hit_sel
        case 'all':
            resp_sel = all_sel
    
    resp_trials = sess_trials[resp_sel]

    sess_avg_rc = ephys_utils.get_psth(resp_trials['abs_click_times'],
                                      resp_trials['cpoke_out'],
                                      rc_window, kernel, resp_trials[['stim_start', 'stim_end']])

    filt_avg = fp_utils.filter_signal(sess_avg_rc['signal_avg'], filt_f, 1/bin_width)
    
    # _, ax = plt.subplots(1,1)
    # ax.axhline(avg_cr, linestyle='dashed', color='black')
    # plot_utils.plot_psth(sess_avg_rc['time'], filt_avg, sess_avg_rc['signal_se'], ax)
    # ax.set_title('Subject {}, Session {} FA RC'.format(sess_trials['subjid'].iloc[0], sess_id))
    
    # _, ax = plt.subplots(1,1)
    # ax.axhline(0, linestyle='dashed', color='black')
    # plot_utils.plot_psth(sess_avg_rc['time'], filt_avg-avg_cr, sess_avg_rc['signal_se'], ax)
    # ax.set_title('Subject {}, Session {} FA RC'.format(sess_trials['subjid'].iloc[0], sess_id))
    
    # plot RCs without smoothing for early/late trials in session as well as long/short trials by nosepoke time
    med_trial = np.median(resp_trials['trial'])
    med_cpoke_dur = np.median(resp_trials['cpoke_dur'])
    
    early_sel = resp_trials['trial'] < med_trial
    late_sel = resp_trials['trial'] > med_trial
    short_sel = resp_trials['cpoke_dur'] < med_cpoke_dur
    long_sel = resp_trials['cpoke_dur'] > med_cpoke_dur

    plot_sels = [[early_sel & short_sel, early_sel & long_sel, late_sel & short_sel, late_sel & long_sel],
                 [early_sel, late_sel], [short_sel, long_sel]]
    plot_labels = [['Early/Short', 'Early/Long', 'late/Short', 'late/Long'], ['Early', 'Late'], ['Short', 'Long']]

    fig, axs = plt.subplots(1,3, figsize=(12,4), layout='constrained', sharey=True)
    for i, (plot_sel, plot_label) in enumerate(zip(plot_sels, plot_labels)):
        ax = axs[i]
        ax.axhline(0, color='black', linestyle='dotted')
        for sel, label in zip(plot_sel, plot_label):
            rc_dict = ephys_utils.get_psth(resp_trials.loc[sel, 'abs_click_times'],
                                      resp_trials.loc[sel, 'cpoke_out'],  # align to poke out
                                      rc_window, kernel, resp_trials.loc[sel, ['stim_start', 'stim_end']])

            filt_sig = fp_utils.filter_signal(rc_dict['signal_avg'], filt_f, 1/bin_width)                          
            plot_utils.plot_psth(rc_dict['time'], filt_sig-avg_cr, error=rc_dict['signal_se'], ax=ax, label=label)

        ax.plot(sess_avg_rc['time'], filt_avg-avg_cr, color='black', linestyle='dashed')
        ax.set_ylabel('Click Rate (Hz)')
        ax.set_xlabel('Time from Response (s)')
        ax.legend()
    fig.suptitle('{} - {}'.format(sess_trials['subjid'].iloc[0], sess_id))
    
# %% Look at all sessions together by subject

bin_width = 0.02
rc_window = np.array([-0.8, 0])
kernel = ephys_utils.get_filter_kernel(filter_type='none', bin_width=bin_width)
filt_f = 5

trial_sel = 'hit' # 'fa' 'hit' 'all'

for subj_id in subj_ids:
    subj_trials = sess_data[sess_data['subjid'] == subj_id]
    
    fa_sel = ((subj_trials['FA'] == 1) & (subj_trials['cpoke_dur'] > 0.5)) & subj_trials['valid']
    hit_sel = (subj_trials['hit'] == 1) & subj_trials['valid']
    all_sel = fa_sel | hit_sel 
    
    # compute average pre-change firing rate for response trials as baseline to subtract from RC kernels
    avg_cr = get_base_cr(subj_trials[all_sel])
    
    match trial_sel:
        case 'fa':
            resp_sel = fa_sel
        case 'hit':
            resp_sel = hit_sel
        case 'all':
            resp_sel = all_sel
    
    resp_trials = subj_trials[resp_sel]

    subj_avg_rc = ephys_utils.get_psth(resp_trials['abs_click_times'],
                                      resp_trials['cpoke_out'],
                                      rc_window, kernel, resp_trials[['stim_start', 'stim_end']])

    filt_avg = fp_utils.filter_signal(subj_avg_rc['signal_avg'], filt_f, 1/bin_width)
    
    # plot RCs without smoothing for early/late trials in session as well as long/short trials by nosepoke time
    med_trial = np.median(resp_trials['trial'])
    med_cpoke_dur = np.median(resp_trials['cpoke_dur'])
    
    early_sel = resp_trials['trial'] < med_trial
    late_sel = resp_trials['trial'] > med_trial
    short_sel = resp_trials['cpoke_dur'] < med_cpoke_dur
    long_sel = resp_trials['cpoke_dur'] > med_cpoke_dur

    plot_sels = [[early_sel & short_sel, early_sel & long_sel, late_sel & short_sel, late_sel & long_sel],
                 [early_sel, late_sel], [short_sel, long_sel]]
    plot_labels = [['Early/Short', 'Early/Long', 'late/Short', 'late/Long'], ['Early', 'Late'], ['Short', 'Long']]

    fig, axs = plt.subplots(1,3, figsize=(12,4), layout='constrained', sharey=True)
    for i, (plot_sel, plot_label) in enumerate(zip(plot_sels, plot_labels)):
        ax = axs[i]
        ax.axhline(0, color='black', linestyle='dotted')
        for sel, label in zip(plot_sel, plot_label):
            rc_dict = ephys_utils.get_psth(resp_trials.loc[sel, 'abs_click_times'],
                                      resp_trials.loc[sel, 'cpoke_out'],  # align to poke out
                                      rc_window, kernel, resp_trials.loc[sel, ['stim_start', 'stim_end']])

            filt_sig = fp_utils.filter_signal(rc_dict['signal_avg'], filt_f, 1/bin_width)                          
            plot_utils.plot_psth(rc_dict['time'], filt_sig-avg_cr, error=rc_dict['signal_se'], ax=ax, label=label)

        ax.plot(subj_avg_rc['time'], filt_avg-avg_cr, color='black', linestyle='dashed')
        ax.set_ylabel('Click Rate (Hz)')
        ax.set_xlabel('Time from Response (s)')
        ax.legend()
    fig.suptitle('{}'.format(subj_id))

# %% Look at more discretized trial lengths/numbers

bin_width = 0.02
rc_window = np.array([-0.8, 0])
kernel = ephys_utils.get_filter_kernel(filter_type='none', bin_width=bin_width)
filt_f = 5

n_bins = 6
disc_col = 'cpoke_dur' # 'cpoke_dur' 'trial'
leg_titles = {'cpoke_dur': 'Cpoke Duration', 'trial': 'Trial Number'}
leg_labels = {'cpoke_dur': '{:.2f}-{:.2f}', 'trial': '{:.0f}-{:.0f}'}

trial_sel = 'fa' # 'fa' 'hit' 'all'

zero_baseline = True

for subj_id in subj_ids:
    subj_trials = sess_data[sess_data['subjid'] == subj_id]
    
    fa_sel = ((subj_trials['FA'] == 1) & (subj_trials['cpoke_dur'] > 0.5)) & subj_trials['valid']
    hit_sel = (subj_trials['hit'] == 1) & subj_trials['valid']
    all_sel = fa_sel | hit_sel 
    
    # compute average pre-change firing rate for response trials as baseline to subtract from RC kernels
    avg_cr = get_base_cr(subj_trials[all_sel])
    
    match trial_sel:
        case 'fa':
            resp_sel = fa_sel
        case 'hit':
            resp_sel = hit_sel
        case 'all':
            resp_sel = all_sel
    
    resp_trials = subj_trials[resp_sel]

    # get percentiles of discretized bin
    disc_bin_edges = np.quantile(resp_trials[disc_col], np.linspace(0,1,n_bins+1))
    
    fig, ax = plt.subplots(1,1)
    ax.axhline(0, color='black', linestyle='dotted')

    for i in range(n_bins):
        sel = (resp_trials[disc_col] >= disc_bin_edges[i]) & (resp_trials[disc_col] <= disc_bin_edges[i+1])
        
        rc_dict = ephys_utils.get_psth(resp_trials.loc[sel, 'abs_click_times'],
                                    resp_trials.loc[sel, 'cpoke_out'],  # align to poke out
                                    rc_window, kernel, resp_trials.loc[sel, ['stim_start', 'stim_end']])

        filt_sig = fp_utils.filter_signal(rc_dict['signal_avg'], filt_f, 1/bin_width) - avg_cr
        
        if zero_baseline:
            peak_start_idx = find_peak_start(filt_sig, rc_dict['time'])
            # use average over 200 ms prior to peak
            avg_start_idx = np.argmin(np.abs(rc_dict['time'] - rc_dict['time'][peak_start_idx] + 0.2))
            base_avg = np.mean(filt_sig[avg_start_idx:peak_start_idx])
            filt_sig -= base_avg

        plot_utils.plot_psth(rc_dict['time'], filt_sig, error=rc_dict['signal_se'], 
            ax=ax, label=leg_labels[disc_col].format(disc_bin_edges[i], disc_bin_edges[i+1]))

    ax.set_ylabel('Change in Click Rate (Hz)')
    ax.set_xlabel('Time from Response (s)')
    ax.legend(title=leg_titles[disc_col])
    fig.suptitle('{}'.format(subj_id))

# %% Use the baseline adjusted session FA RC kernel as the click rate kernel to perform regression with firing rates over time in a trial

bin_width = 0.02
rc_window = np.array([-0.8, 0])
rc_kernel = ephys_utils.get_filter_kernel(filter_type='none', bin_width=bin_width)
fr_kernel = ephys_utils.get_filter_kernel(filter_type='half_gauss', width=0.4, bin_width=bin_width)
filt_f = 5 # the low-pass filter cutoff to make the click rate kernel
pre_peak_start_window = 0.1
    
n_trial_bins = 1 # number of bins to split the trials from each session into for decoding
cpoke_dur_step = 0.3 # width of bins to split the poke durations into for decoding
use_all_units = True

pre_move_buffer = 0.1
trial_start_buffer = 0.1

all_fit_results = []

n_splits = 20
#cv = skl_ms.GroupKFold(n_splits)
cv = skl_ms.ShuffleSplit(n_splits, test_size=0.1)
# perform linear regression of firing rates on click rates using all units
regr = linear_model.LinearRegression()
scores = {'r2': make_scorer(r2_score), 'rmse': make_scorer(root_mean_squared_error)}

def fit_model(x, y, groups):
    if n_splits > 1:
        # catch warnings if split method doesn't use groups
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cv_scores = skl_ms.cross_validate(regr, x, y, groups=groups, cv=cv, scoring=scores)
            results = {'r2': cv_scores['test_r2'], 'rmse': cv_scores['test_rmse']}
    else:
        regr.fit(x, y)
        pred_y = regr.predict(x)
        results = {'r2': r2_score(y, pred_y), 'rmse': root_mean_squared_error(y, pred_y)}
    return results    

for sess_id in utils.flatten(sess_ids): #[47622, 48261]: #, 49702, 49636]: # [48261]: # 
    sess_trials = sess_data[sess_data['sessid'] == sess_id]
    fa_sel = ((sess_trials['FA'] == 1) & (sess_trials['cpoke_dur'] > 0.5)) & sess_trials['valid']
    resp_sel = ((sess_trials['hit'] == 1) & sess_trials['valid']) | fa_sel
    resp_trials = sess_trials[resp_sel]
    fa_trials = sess_trials[fa_sel]

    sess_avg_rc = ephys_utils.get_psth(fa_trials['abs_click_times'],
                                       fa_trials['cpoke_out'],
                                       rc_window, rc_kernel, fa_trials[['stim_start', 'stim_end']])

    filt_avg = fp_utils.filter_signal(sess_avg_rc['signal_avg'], filt_f, 1/bin_width)

    peak_start_idx = find_peak_start(filt_avg, sess_avg_rc['time'])
    # use average over some amount of time prior to peak
    avg_start_idx = np.argmin(np.abs(sess_avg_rc['time'] - sess_avg_rc['time'][peak_start_idx] + pre_peak_start_window))
    base_avg = np.mean(filt_avg[avg_start_idx:peak_start_idx])
    filt_avg -= base_avg

    # 0 out negative values and normalize to sum to 1
    cr_kernel_weights = filt_avg[avg_start_idx:]
    cr_kernel_weights[cr_kernel_weights < 0] = 0
    cr_kernel_weights = cr_kernel_weights/np.sum(cr_kernel_weights)
     # make the kernel dictionary to work with the ephys methods
    cr_kernel = {'weights': np.flip(cr_kernel_weights),
                 'bin_width': bin_width,
                 'center_idx': 0}
    
    # end the stimulus period at cpoke out or stimulus end, whichever comes first
    stim_start = sess_trials['stim_start'].to_numpy()
    stim_end = sess_trials['stim_end'].to_numpy()
    cpoke_out = sess_trials['cpoke_out'].to_numpy()
    
    stim_cpoke_end = stim_end.copy()
    end_late_sel = stim_end > cpoke_out
    stim_cpoke_end[end_late_sel] = cpoke_out[end_late_sel]
    stim_cpoke_end = stim_cpoke_end - pre_move_buffer
    
    rel_stim_cpoke_end = stim_cpoke_end - stim_start
    
    # compute the smoothed trial-by-trial click rate
    trial_crs = pd.Series([ephys_utils.get_smoothed_firing_rate(
                    resp_trials['rel_click_times'].iloc[i],
                    cr_kernel, trial_start_buffer,
                    rel_stim_cpoke_end[resp_sel][i])[0][:,None]
                for i in range(len(resp_trials))])
    
    # look through smoothed click rates
    # for i in range(10):
    #     clicks = resp_trials['abs_click_times'].iloc[i]
    #     smoothed_cr, t = ephys_utils.get_smoothed_firing_rate(
    #                         clicks, cr_kernel, resp_trials['stim_start'].iloc[i],
    #                         resp_trials['cpoke_out'].iloc[i])
    #     _, ax = plt.subplots(1,1)
    #     ax.plot(t, smoothed_cr)
    #     ax.vlines(clicks[clicks < t[-1]], 0, 1, color='black')

    # get firing rate matrix for the stimulus period
    sess_units = unit_data[unit_data['sessid'] == sess_id]
    trial_start_ts = sess_units['trial_start_timestamps'].iloc[0][0:-1]
    
    # first filter the units whose number of spikes is at least 1 Hz for nosepoke periods
    abs_stim_start = trial_start_ts + stim_start
    abs_stim_cpoke_end = trial_start_ts + stim_cpoke_end

    poke_frs = ephys_utils.get_avg_period_fr(sess_units['spike_timestamps'], np.hstack((abs_stim_start[:,None], abs_stim_cpoke_end[:,None])), resp_sel)
    
    sess_units = sess_units[poke_frs > 1]

    # compute the smoothed firing rate matrices by trial
    rel_poke_bounds = np.hstack((np.full_like(rel_stim_cpoke_end, trial_start_buffer)[:,None], rel_stim_cpoke_end[:,None]))

    fr_mats = ephys_utils.get_fr_matrix_by_trial(sess_units['spike_timestamps'], abs_stim_start, fr_kernel, rel_poke_bounds, resp_sel)[0]

    poke_dur_bin_edges = np.arange(trial_start_buffer, np.max(resp_trials['cpoke_dur'])-pre_move_buffer+cpoke_dur_step, cpoke_dur_step)
    
    fit_results = []
    
    # get all trials together as well as split into bins for comparison
    if n_trial_bins > 1:
        trial_bins_to_use = [1, n_trial_bins]
    else:
        trial_bins_to_use = [1]
        
    for trial_bins in trial_bins_to_use:
        trial_bin_edges = np.quantile(resp_trials['trial'], np.linspace(0,1,trial_bins+1))
        
        for i in range(trial_bins):
            trial_sel = (resp_trials['trial'] >= trial_bin_edges[i]) & (resp_trials['trial'] <= trial_bin_edges[i+1])
            trial_sel = trial_sel.to_numpy()
            if trial_bins == 1:
                trials_label = 'All'
            else:
                trials_label = '{:.0f}-{:.0f}'.format(trial_bin_edges[i], trial_bin_edges[i+1])
            
            sub_trial_crs = trial_crs[trial_sel]
            sub_trial_frs = fr_mats[trial_sel]
            
            # look at the whole trial as well
            for whole_trial in [True, False]:
                if whole_trial:
                    dur_label = 'All'
                    stacked_trial_crs = np.vstack(sub_trial_crs)
                    trial_nums = np.vstack([np.full((len(crs), 1), i) for i, crs in enumerate(sub_trial_crs)])

                    if use_all_units:
                        
                        all_frs = np.vstack([m for m in sub_trial_frs])
                        
                        fit_scores = fit_model(all_frs, stacked_trial_crs, trial_nums)

                        fit_results.append({'trials': trials_label,
                                            'time': dur_label,
                                            'n_trials': np.sum(trial_sel),
                                            'r2': np.mean(fit_scores['r2']),
                                            'rmse': np.mean(fit_scores['rmse'])})
                    else:
                        unit_r2s = []
                        for k in range(len(sess_units)):

                            unit_fr = np.vstack([m[:,k][:,None] for m in sub_trial_frs])
                            regr.fit(unit_fr, stacked_trial_crs)
                            r2 = regr.score(unit_fr, stacked_trial_crs)
                            rmse = root_mean_squared_error(stacked_trial_crs, regr.predict(unit_fr))
                            unit_r2s.append({'i': i,
                                             'unitid': sess_units['unitid'].iloc[i],
                                             'r2': r2,
                                             'rmse': rmse})
                            
                        unit_r2s = pd.DataFrame(unit_r2s).sort_values('r2', ascending=False)
                        fit_results.append({'trials': trials_label,
                                            'time': dur_label,
                                            'n_trials': np.sum(trial_sel),
                                            'results': unit_r2s})
                else:
                    # go bin by bin aligned to start (forward = True) and end of poke
                    for forward in [True, False]:
                        for j in range(len(poke_dur_bin_edges)-1):
                            dur_bin_start_idx = int((poke_dur_bin_edges[j]-trial_start_buffer)/bin_width)
                            dur_bin_end_idx = int((poke_dur_bin_edges[j+1]-trial_start_buffer)/bin_width)
                            dur_trial_sel = resp_trials[trial_sel]['cpoke_dur']-pre_move_buffer >= poke_dur_bin_edges[j]
                            dur_trial_sel = dur_trial_sel.to_numpy()
                            
                            if np.sum(dur_trial_sel) < n_splits:
                                continue
                            
                            if forward:
                                sub_crs = [cr[dur_bin_start_idx:dur_bin_end_idx] if len(cr) > dur_bin_end_idx 
                                           else cr[dur_bin_start_idx:] for cr in sub_trial_crs[dur_trial_sel]]
                                alignment = 'Cpoke In'
                            else:
                                sub_crs = [np.flip(cr)[dur_bin_start_idx:dur_bin_end_idx] if len(cr) > dur_bin_end_idx 
                                           else np.flip(cr)[dur_bin_start_idx:] for cr in sub_trial_crs[dur_trial_sel]]
                                alignment = 'Cpoke Out'
                                
                            dur_label = '{:.1f}-{:.1f}'.format(poke_dur_bin_edges[j], poke_dur_bin_edges[j+1])
            
                            stacked_trial_crs = np.vstack(sub_crs)
                            trial_nums = np.vstack([np.full((len(crs), 1), i) for i, crs in enumerate(sub_crs)])
                            trial_idxs = np.cumsum([len(cr) for cr in sub_crs])
                            
                            if use_all_units:
                                if forward:
                                    all_frs = np.vstack([m[dur_bin_start_idx:dur_bin_end_idx,:] if m.shape[0] > dur_bin_end_idx 
                                                         else m[dur_bin_start_idx:,:] for m in sub_trial_frs[dur_trial_sel]])
                                else:
                                    all_frs = np.vstack([np.flip(m, axis=0)[dur_bin_start_idx:dur_bin_end_idx,:] if m.shape[0] > dur_bin_end_idx 
                                                         else np.flip(m, axis=0)[dur_bin_start_idx:,:] for m in sub_trial_frs[dur_trial_sel]])
            
                                fit_scores = fit_model(all_frs, stacked_trial_crs, trial_nums)

                                fit_results.append({'trials': trials_label,
                                                    'time': dur_label,
                                                    'alignment': alignment,
                                                    'n_trials': np.sum(dur_trial_sel),
                                                    'r2': np.mean(fit_scores['r2']),
                                                    'rmse': np.mean(fit_scores['rmse'])})
                                
                            else:
                                unit_r2s = []
                                for k in range(len(sess_units)):
                                    if forward:
                                        unit_fr = np.vstack([m[dur_bin_start_idx:dur_bin_end_idx,k][:,None] if m.shape[0] > dur_bin_end_idx 
                                                             else m[dur_bin_start_idx:,k][:,None] for m in sub_trial_frs[dur_trial_sel]])
                                    else:
                                        unit_fr = np.vstack([np.flip(m, axis=0)[dur_bin_start_idx:dur_bin_end_idx,k][:,None] if m.shape[0] > dur_bin_end_idx 
                                                             else np.flip(m, axis=0)[dur_bin_start_idx:,k][:,None] for m in sub_trial_frs[dur_trial_sel]])
                                    
                                    regr.fit(unit_fr, stacked_trial_crs)
                                    r2 = regr.score(unit_fr, stacked_trial_crs)
                                    rmse = root_mean_squared_error(stacked_trial_crs, regr.predict(unit_fr))
                                    unit_r2s.append({'i': k,
                                                     'unitid': sess_units['unitid'].iloc[k],
                                                     'r2': r2,
                                                     'rmse': rmse})
                                    
                                unit_r2s = pd.DataFrame(unit_r2s).sort_values('r2', ascending=False)
                                fit_results.append({'trials': trials_label,
                                                    'time': dur_label,
                                                    'alignment': alignment,
                                                    'n_trials': np.sum(dur_trial_sel),
                                                    'results': unit_r2s})
                            
                                # # plot all R2 values
                                # _, ax = plt.subplots(1,1)
                                # ax.hist(unit_r2s['r2'])
                                # ax.set_xlabel('R2 Values')
                                # ax.set_ylabel('Counts')
                                # ax.set_title('{} - {}, Unit {}, Trials {}, Time {} from {}'.format(
                                #     sess_trials['subjid'].iloc[0], sess_id, unit_r2s.iloc[i]['unitid'], trials_label, dur_label, alignment))
                            
                                # # plot top N R2 values
                                # for i in range(3):
                                #     unit_idx = int(unit_r2s.iloc[i]['i'])
                                #     if forward:
                                #         unit_fr = np.vstack([m[dur_bin_start_idx:dur_bin_end_idx,unit_idx][:,None] if m.shape[0] > dur_bin_end_idx 
                                #                              else m[dur_bin_start_idx:,unit_idx][:,None] for m in sub_trial_frs[dur_trial_sel]])
                                #     else:
                                #         unit_fr = np.vstack([np.flip(m, axis=0)[dur_bin_start_idx:dur_bin_end_idx,unit_idx][:,None] if m.shape[0] > dur_bin_end_idx 
                                #                              else np.flip(m, axis=0)[dur_bin_start_idx:,unit_idx][:,None] for m in sub_trial_frs[dur_trial_sel]])
                                        
                                #     regr.fit(unit_fr, stacked_trial_crs)
                                    
                                #     fig, axs = plt.subplots(1,2)
                                #     fig.suptitle('{} - {}, Unit {}, Trials {}, Time {} from {}'.format(
                                #         sess_trials['subjid'].iloc[0], sess_id, unit_r2s.iloc[i]['unitid'], trials_label, dur_label, alignment))
                                    
                                #     fit_x = np.linspace(np.min(unit_fr), np.max(unit_fr), 100)[:,None]
                                #     fit_y = regr.predict(fit_x)
                            
                                #     ax = axs[0]
                                #     ax.scatter(unit_fr, stacked_trial_crs)
                                #     ax.plot(fit_x, fit_y, linestyle='dashed', linewidth=2, color='C1')
                                #     ax.set_xlabel('Unit Firing Rate (Hz)')
                                #     ax.set_ylabel('Estimated Click Rate (Hz)')
                                    
                                #     ax = axs[1]
                                #     plot_utils.plot_dashlines(trial_idxs, ax=ax, label='_')
                                #     ax.plot(stacked_trial_crs/np.std(stacked_trial_crs), label='click rate')
                                #     ax.plot(unit_fr/np.std(unit_fr), label='firing rate')
                                #     ax.set_ylabel('Rate')
                                    
                                #     ax.legend()
                                    
                                    
                                    
    if use_all_units:
        fit_results = pd.DataFrame(fit_results)
    
    all_fit_results.append({'subjid': sess_trials['subjid'].iloc[0],
                            'sessid': sess_id,
                            'n_units': len(sess_units),
                            'results': fit_results,
                            'kernel': {'weights': cr_kernel_weights, 'time': sess_avg_rc['time'][avg_start_idx:]}})
    
save_path = path.join(utils.get_user_home(), 'db_data', 'click_rate_regression_time.pkl')

with open(save_path, 'wb') as f:
    pickle.dump(all_fit_results, f)

for fit_result_data in all_fit_results:
    fit_results = fit_result_data['results']
    kernel = fit_result_data['kernel']
    
    _, ax = plt.subplots(1,1)
    ax.plot(kernel['time'], kernel['weights'])
    ax.set_title('{} - {} FA RC Click Rate Kernel'.format(fit_result_data['subjid'], fit_result_data['sessid']))
        
    for plot_col, col_label in zip(['r2', 'rmse'], ['R2', 'RMSE']): #[['rmse', 'RMSE']]: #
        
        fig, axs = plt.subplots(2,2, layout='constrained', figsize=(10,7), sharey='row')
        fig.suptitle('{} - {}, {} Units'.format(fit_result_data['subjid'], fit_result_data['sessid'], fit_result_data['n_units']))
        
        all_trials = fit_results[(fit_results['trials'] == 'All') & (fit_results['time'] == 'All')]
        all_time_trials = fit_results[(fit_results['trials'] != 'All') & (fit_results['time'] == 'All')]
        trial_cats = sorted(np.unique(all_time_trials['trials']), key=lambda x: int(x.split('-')[0]))
        trial_cat_order = trial_cats.copy()
        trial_cat_order.append('All')
        
        for i, forward in enumerate([True, False]):
            for j, (col, ylabel) in enumerate(zip([plot_col, 'n_trials'], [col_label, '# Trials'])):
            
                ax = axs[j,i]
                ax.axhline(all_trials[col].iloc[0], linestyle='dashed', color='black', label='_', alpha=0.7)
                
                trial_colors = []
                for j, trials in enumerate(trial_cats):
                    trial_color = 'C'+str(j)
                    ax.axhline(all_time_trials.loc[all_time_trials['trials'] == trials, col].iloc[0], linestyle='dashed', 
                               color=trial_color, label='_', alpha=0.7)
                    trial_colors.append(trial_color)
                    
                trial_colors.append('black')
                
                trial_colors = sb.color_palette(trial_colors)
                
                if forward:
                    alignment = 'Cpoke In'
                else:
                    alignment = 'Cpoke Out'
                    
                align_results = fit_results[fit_results['alignment'] == alignment]
                    
                lines = sb.pointplot(align_results, x='time', y=col, hue='trials', hue_order=trial_cat_order, palette=trial_colors, ax=ax, alpha=0.7)
                
                ax.set_title('Aligned to {}'.format(alignment))
                ax.set_ylabel(ylabel)
                ax.set_xlabel('Time from {}'.format(alignment))
                
                if not forward:
                    ax.invert_xaxis()
                
                if col == 'r2':
                    ax.set_ylim(0, 1)
                    
                ax.tick_params(axis='x', labelrotation=45)

# %% Compare click rate decoding using FA RC kernel versus other click rate kernel shapes and widths

bin_width = 0.02
rc_window = np.array([-0.8, 0])
filt_f = 5 # the low-pass filter cutoff to make the click rate kernel
pre_peak_start_window = 0.1
rc_kernel = ephys_utils.get_filter_kernel(filter_type='none', bin_width=bin_width)
fr_kernel = ephys_utils.get_filter_kernel(filter_type='half_gauss', width=0.4, bin_width=bin_width)
comp_kernel_types = ['half_gauss']
comp_kernel_widths = np.arange(0.2, 10, 0.2)
#comp_kernel_widths = [0.5, 1.5, 3]
ephys_offsets = np.arange(-1, 1, 0.05)
#ephys_offsets = [-0.2, 0, 0.2]
recalculate = False
save_results = True

name = 'click_rate_regression_kernel_comp'

n_shuffles = 100

regions = ['FOF', 'ADS', 'PPC']

pre_move_buffer = 0.1
trial_start_buffer = 0.3
normalize_crs = False
shuffle_type = 'avg'

n_splits = 1
#cv = skl_ms.GroupKFold(n_splits)
cv = skl_ms.ShuffleSplit(n_splits, test_size=0.1)
# perform linear regression of firing rates on click rates using all units
regr = linear_model.LinearRegression()
scores = {'r2': make_scorer(r2_score), 'rmse': make_scorer(root_mean_squared_error)}

def fit_model(x, y, groups):
    if n_splits > 1:
        # catch warnings if split method doesn't use groups
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cv_scores = skl_ms.cross_validate(regr, x, y, groups=groups, cv=cv, scoring=scores)
            results = {'r2': cv_scores['test_r2'], 'rmse': cv_scores['test_rmse'], 'pred_y': regr.fit(x, y).predict(x)}
    else:
        regr.fit(x, y)
        pred_y = regr.predict(x)
        results = {'r2': r2_score(y, pred_y), 'rmse': root_mean_squared_error(y, pred_y), 'pred_y': pred_y}
    return results

rng = np.random.default_rng()

def shuffle_crs(cr_list, shuffle_type):
    new_crs = []
    match shuffle_type:
        case 'circle':
            # circularly permute each click rate array in the list at a random location
            #rand_pct = rng.random(len(cr_list))
            for crs in cr_list:
                if len(crs) > 2:
                    new_start_idx = rng.integers(1, len(crs))
                    #new_start_idx = int(len(crs)*pct)
                    new_cr = np.concatenate((crs[new_start_idx:], crs[:new_start_idx]))
                else:
                    new_cr = crs
                new_crs.append(new_cr)
        case 'shuffle_replace':
            # shuffle click rates with replacement on a trial-by-trial basis
            for crs in cr_list:
                new_cr = rng.choice(crs, size=np.max(crs.shape), replace=True)
                new_crs.append(new_cr)
        case 'shuffle_noreplace':
            # shuffle click rates without replacement on a trial-by-trial basis
            for crs in cr_list:
                new_cr = rng.choice(crs, size=np.max(crs.shape), replace=False)
                new_crs.append(new_cr)
        case 'trial_avg':
            for crs in cr_list:
                new_cr = np.full_like(crs, np.mean(crs))
                new_crs.append(new_cr)
    return new_crs

save_path = path.join(utils.get_user_home(), 'db_data', name+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        all_fit_results = pickle.load(f)
else:
    all_fit_results = pd.DataFrame(columns = ['subjid', 'sessid', 'region', 'kernel', 
                                              'n_trials', 'n_units', 'r2', 'rmse', 
                                              'null_r2', 'null_rmse', 'p_r2', 'p_rmse'])

for sess_id in [47622]: #, 48261, 67430]: # 47885, 47928, 67430]: # [48261]: # utils.flatten(sess_ids): #

    all_start = time.perf_counter()
    print('Starting session {}'.format(sess_id))
    
    sess_trials = sess_data[sess_data['sessid'] == sess_id]
    fa_sel = ((sess_trials['FA'] == 1) & (sess_trials['cpoke_dur'] > 0.5)) & sess_trials['valid']
    resp_sel = ((sess_trials['hit'] == 1) & sess_trials['valid']) | fa_sel
    resp_trials = sess_trials[resp_sel]
    fa_trials = sess_trials[fa_sel]

    sess_avg_rc = ephys_utils.get_psth(fa_trials['abs_click_times'],
                                       fa_trials['cpoke_out'],
                                       rc_window, rc_kernel, fa_trials[['stim_start', 'cpoke_out']],
                                       align='end')

    filt_avg = fp_utils.filter_signal(sess_avg_rc['signal_avg'], filt_f, 1/bin_width)

    peak_start_idx = find_peak_start(filt_avg, sess_avg_rc['time'])
    # use average over some amount of time prior to peak
    avg_start_idx = np.argmin(np.abs(sess_avg_rc['time'] - sess_avg_rc['time'][peak_start_idx] + pre_peak_start_window))
    base_avg = np.mean(filt_avg[avg_start_idx:peak_start_idx])
    filt_avg -= base_avg

    # 0 out negative values and normalize to sum to 1
    cr_kernel_weights = filt_avg[avg_start_idx:]
    cr_kernel_weights[cr_kernel_weights < 0] = 0
    cr_kernel_weights = cr_kernel_weights/np.sum(cr_kernel_weights)
     # make the kernel dictionary to work with the ephys methods
    farc_kernel = {'weights': np.flip(cr_kernel_weights),
                   'bin_width': bin_width,
                   'center_idx': 0}
    
    # end the stimulus period at cpoke out or stimulus end, whichever comes first
    stim_start = sess_trials['stim_start'].to_numpy()
    stim_end = sess_trials['stim_end'].to_numpy()
    cpoke_out = sess_trials['cpoke_out'].to_numpy()
    
    stim_cpoke_end = stim_end.copy()
    end_late_sel = stim_end > cpoke_out
    stim_cpoke_end[end_late_sel] = cpoke_out[end_late_sel]
    stim_cpoke_end = stim_cpoke_end - pre_move_buffer
    
    rel_stim_cpoke_end = stim_cpoke_end - stim_start
    rel_poke_bounds = np.hstack((np.full_like(rel_stim_cpoke_end, trial_start_buffer)[:,None], rel_stim_cpoke_end[:,None]))

    # get firing rates for the stimulus period
    sess_units = unit_data[unit_data['sessid'] == sess_id]
    trial_start_ts = sess_units['trial_start_timestamps'].iloc[0][0:-1]
    
    # first filter the units whose number of spikes is at least 1 Hz for nosepoke periods
    abs_stim_start = trial_start_ts + stim_start
    abs_stim_cpoke_end = trial_start_ts + stim_cpoke_end

    poke_frs = ephys_utils.get_avg_period_fr(sess_units['spike_timestamps'], np.hstack((abs_stim_start[:,None], abs_stim_cpoke_end[:,None])))
    
    sess_units = sess_units[poke_frs > 1]
    
    # decode separately by region
    sess_regions = np.intersect1d(np.unique(sess_units['region']), regions)
    
    kernel_types = ['FARC'] + comp_kernel_types
    kernel_widths = {'FARC': [0]}
    kernel_widths.update({t: comp_kernel_widths for t in comp_kernel_types})

    for region in sess_regions:
        region_start = time.perf_counter()
        region_units = sess_units[sess_units['region'] == region]
        
        # check if we need to finish analyzing this session/region
        existing_rows = all_fit_results[(all_fit_results['sessid'] == sess_id) & (all_fit_results['region'] == region)]
        new_kernel_types = np.setdiff1d(kernel_types, existing_rows['kernel'].to_list())
        if len(new_kernel_types) == 0: # and 
        #     np.all(~np.isnan(existing_rows[existing_rows['kernel'] == comp_kernel_type]['r2'].iloc[0])) and
        #     existing_rows[existing_rows['kernel'] == comp_kernel_type]['r2'].iloc[0].shape == (len(kernel_widths[comp_kernel_type]), len(ephys_offsets))):
            
            continue
            
        r2_mats = {t: np.full((len(kernel_widths[t]), len(ephys_offsets)), np.nan) for t in new_kernel_types}
        rmse_mats = cp.deepcopy(r2_mats)
        null_r2_mats = cp.deepcopy(r2_mats)
        null_rmse_mats = cp.deepcopy(r2_mats)
        p_r2_mats = cp.deepcopy(r2_mats)
        p_rmse_mats = cp.deepcopy(r2_mats)
        
        for i, ephys_offset in enumerate(ephys_offsets):
            
            offset_start = time.perf_counter()
            
            # compute the smoothed firing rate matrices by trial
            fr_mats = ephys_utils.get_fr_matrix_by_trial(region_units['spike_timestamps'], abs_stim_start + ephys_offset, 
                                                         fr_kernel, rel_poke_bounds, resp_sel)[0]
            stacked_frs = np.vstack(fr_mats)
    
            # compute the smoothed trial-by-trial click rate for different kernels
            for kernel_type in new_kernel_types:
                match kernel_type:
                    case 'FARC':
                        kern_method = lambda x: farc_kernel
                    case _:               
                        kern_method = lambda x: ephys_utils.get_filter_kernel(filter_type=kernel_type, width=x, bin_width=bin_width)
            
                for j, kern_width in enumerate(kernel_widths[kernel_type]):
                    
                    width_start = time.perf_counter()
                    
                    cr_kernel = kern_method(kern_width)
                    
                    trial_crs = [ephys_utils.get_smoothed_firing_rate(
                                        resp_trials['rel_click_times'].iloc[i],
                                        cr_kernel, trial_start_buffer,
                                        rel_stim_cpoke_end[resp_sel][i])[0][:,None]
                                     for i in range(len(resp_trials))]
        
                    stacked_trial_crs = np.vstack(trial_crs)
                    trial_idxs = np.cumsum([len(cr) for cr in trial_crs])
                    if normalize_crs:
                        mean_cr = np.mean(stacked_trial_crs)
                        std_cr = np.std(stacked_trial_crs)
                        stacked_trial_crs = (stacked_trial_crs - mean_cr)/std_cr
                        trial_crs = [(crs - mean_cr)/std_cr for crs in trial_crs]
                        
                    trial_nums = np.vstack([np.full((len(crs), 1), i) for i, crs in enumerate(trial_crs)])
                    
                    shuffled_results = []
                    # circle_results = []
                    # shuffle_replace_results = []
                    # shuffle_noreplace_results = []
                    if shuffle_type == 'avg':
                        mean_cr = np.full_like(stacked_trial_crs, np.mean(stacked_trial_crs))
                        shuffled_results = [{'r2': r2_score(stacked_trial_crs, mean_cr), 'rmse': root_mean_squared_error(stacked_trial_crs, mean_cr)}]
                    else:
                        for k in range(n_shuffles):
                            shuffled_crs = np.vstack(shuffle_crs(trial_crs, shuffle_type))
                            fit_scores = fit_model(stacked_frs, shuffled_crs, trial_nums)
                            shuffled_results.append({'r2': np.mean(fit_scores['r2']), 'rmse': np.mean(fit_scores['rmse'])})
                            
                            # shuffled_crs = np.vstack(shuffle_crs(trial_crs, 'shuffle_replace'))
                            # fit_scores = fit_model(stacked_frs, shuffled_crs, trial_nums)
                            # shuffle_replace_results.append({'r2': np.mean(fit_scores['r2']), 'rmse': np.mean(fit_scores['rmse'])})
                            
                            # shuffled_crs = np.vstack(shuffle_crs(trial_crs, 'shuffle_noreplace'))
                            # fit_scores = fit_model(stacked_frs, shuffled_crs, trial_nums)
                            # shuffle_noreplace_results.append({'r2': np.mean(fit_scores['r2']), 'rmse': np.mean(fit_scores['rmse'])})

                    shuffled_results = pd.DataFrame(shuffled_results)
                    # circle_avg = circle_results.agg(np.mean)
                    # shuffle_replace_results = pd.DataFrame(shuffle_replace_results)
                    # shuffle_replace_avg = shuffle_replace_results.agg(np.mean)
                    # shuffle_noreplace_results = pd.DataFrame(shuffle_noreplace_results)
                    # shuffle_noreplace_avg = shuffle_noreplace_results.agg(np.mean)
                    
                    # mean_cr = np.full_like(stacked_trial_crs, np.mean(stacked_trial_crs))
                    # mean_null_results = {'r2': r2_score(stacked_trial_crs, mean_cr), 'rmse': root_mean_squared_error(stacked_trial_crs, mean_cr)}
                    
                    # trial_mean_crs = np.vstack(shuffle_crs(trial_crs, 'mean'))
                    # trial_mean_null_results = {'r2': r2_score(stacked_trial_crs, trial_mean_crs), 'rmse': root_mean_squared_error(stacked_trial_crs, trial_mean_crs)}
                    
                    fit_scores = fit_model(stacked_frs, stacked_trial_crs, trial_nums)
                    
                    # all_result_comparison = {'model': fit_scores,
                    #                          'circle': circle_avg,
                    #                          'shuffle_replace': shuffle_replace_avg,
                    #                          'shuffle_noreplace': shuffle_noreplace_avg,
                    #                          'mean': mean_null_results,
                    #                          'trial_mean': trial_mean_null_results}
                    
                    # _, ax = plt.subplots(1,1)
                    # ax.scatter(stacked_trial_crs, fit_scores['pred_y'])
                    # ax.set_xlabel('True CRs (Hz)')
                    # ax.set_ylabel('Predicted CRs (Hz)')
                    # ax.set_title('{} {} {} (width = {:.2f}, offset = {:.2f})'.format(sess_id, region, kernel_type, kern_width, ephys_offset))
                    
                    # regr.fit(stacked_frs, stacked_trial_crs)
                    # reg_v = regr.coef_.T
                    # reg_v = reg_v / np.linalg.norm(reg_v)
                    # _, ax = plt.subplots(1,1)
                    # plot_utils.plot_dashlines(trial_idxs, ax=ax, label='_')
                    # ax.plot(utils.z_score(stacked_trial_crs), label='CRs')
                    # ax.plot(utils.z_score(stacked_frs @ reg_v), label='Projected FRs')
                    # ax.set_ylabel('Z-scored CRs or Projected FRs')
                    # ax.set_title('{} {} {} (width = {:.2f}, offset = {:.2f})'.format(sess_id, region, kernel_type, kern_width, ephys_offset))
                    # ax.legend()
                    
                    r2 = np.mean(fit_scores['r2'])
                    rmse = np.mean(fit_scores['rmse'])
                    
                    r2_mats[kernel_type][j,i] = r2
                    rmse_mats[kernel_type][j,i] = rmse
                    null_r2_mats[kernel_type][j,i] = np.mean(shuffled_results['r2'])
                    null_rmse_mats[kernel_type][j,i] = np.mean(shuffled_results['rmse'])
                    p_r2_mats[kernel_type][j,i] = np.sum(shuffled_results['r2'] > r2)/n_shuffles
                    p_rmse_mats[kernel_type][j,i] = np.sum(shuffled_results['rmse'] < rmse)/n_shuffles
                    
                    print('      Processed {} kernel width {:.2f} in {:.1f} s'.format(kernel_type, kern_width, time.perf_counter()-width_start))

            print('    Processed ephys offset {:.2f} in {:.1f} s'.format(ephys_offset, time.perf_counter()-offset_start))

        for kernel_type in new_kernel_types:
            all_fit_results = pd.concat((all_fit_results, 
                                        pd.DataFrame([{'subjid': sess_trials['subjid'].iloc[0],
                                                     'sessid': sess_id,
                                                     'region': region,
                                                     'kernel': kernel_type,
                                                     'n_trials': len(resp_trials),
                                                     'n_units': len(region_units),
                                                     'r2': r2_mats[kernel_type],
                                                     'rmse': rmse_mats[kernel_type],
                                                     'null_r2': null_r2_mats[kernel_type],
                                                     'null_rmse': null_rmse_mats[kernel_type],
                                                     'p_r2': p_r2_mats[kernel_type],
                                                     'p_rmse': p_rmse_mats[kernel_type],
                                                     'kernel_widths': kernel_widths[kernel_type],
                                                     'ephys_offsets': ephys_offsets,
                                                     'farc_kernel': {'weights': cr_kernel_weights, 'time': sess_avg_rc['time'][avg_start_idx:]}}])))
            
        print('  Processed region {} in {:.1f} s'.format(region, time.perf_counter()-region_start))
   
    print('Processed session {} in {:.1f} s'.format(sess_id, time.perf_counter()-all_start))

if save_results:
    with open(save_path, 'wb') as f:
        pickle.dump(all_fit_results, f)

# %%        
other_kern = 'half_gauss'
kern_widths = [0.08, 0.13]

_, axs = plt.subplots(2, 2, figsize=(10,6), sharey=True, sharex=True, layout='constrained')

for i, sess_id in enumerate([47622, 67430]):
    farc_kernel_row = all_fit_results[(all_fit_results['sessid'] == sess_id) & (all_fit_results['kernel'] == 'FARC')].iloc[0]
    farc_kernel = farc_kernel_row['farc_kernel']
    #_, ax = plt.subplots(1,1)
    ax = axs[0,i]
    ax.plot(np.flip(-farc_kernel['time']), np.flip(farc_kernel['weights']), label='FA RC')
    ax.set_title('{} FARC Kernel'.format(farc_kernel_row['subjid']))
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.xaxis.set_tick_params(which='both', labelbottom=True)
    
for i, width in enumerate(kern_widths):
    other_kernel = ephys_utils.get_filter_kernel(width*4, other_kern, 0.02)
    #_, ax = plt.subplots(1,1)
    ax = axs[1,i]
    ax.plot(other_kernel['t'], other_kernel['weights'])
    ax.set_title('Kernel Width: {:.0f} ms'.format(width*100))
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.xaxis.set_tick_params(which='both', labelbottom=True)

# %% Plot kernel comparison decoding results

plot_types = ['vals', 'null', 'diff', 'p_vals']
metric_cols = ['r2', 'rmse']
metric_labels = ['R2', 'RMSE']
cmap = 'jet'

for sess_id in np.unique(all_fit_results['sessid']):
    sess_results = all_fit_results[all_fit_results['sessid'] == sess_id]
    subj_id = sess_results['subjid'].iloc[0]
    
    farc_kernel = sess_results['farc_kernel'].iloc[0]
    _, ax = plt.subplots(1,1)
    ax.plot(farc_kernel['time'], farc_kernel['weights'])
    ax.set_title('Subject {}, Session {} FA RC Click Rate Kernel'.format(subj_id, sess_id))
    
    for plot_type in plot_types:
        for region in np.unique(sess_results['region']):
            region_results = sess_results[sess_results['region'] == region]
            
            match plot_type:
                case 'vals':
                    title = 'Model Fit Values'
                case 'null':
                    title = 'Null Model Fit Values'
                case 'diff':
                    title = 'Diff (Model - Null)'
                case 'p_vals':
                    title = '% Null Better Than Model'
            
            kern_types = region_results['kernel'].unique()
            n_kerns = len(kern_types)
            row_heights = [1]+[3 for i in range((n_kerns-1)*2)]
            fig, axs = plt.subplots(len(row_heights), 2, layout='constrained', figsize=(10, np.sum(row_heights)), height_ratios=row_heights)
            fig.suptitle('Subject {}, Session {} (n={}), {} (n={}): {}'.format(subj_id, sess_id, region_results['n_trials'].iloc[0], region, region_results['n_units'].iloc[0], title))
            
            # compute max and min values for each metric so scales are the same down the column
            metric_lims = {}
            for metric in metric_cols:
                match plot_type:
                    case 'vals':
                        metric_lims[metric] = {'max': np.max(region_results[metric].apply(np.max)),
                                               'min': np.min(region_results[metric].apply(np.min))}
                        get_vals = lambda x, m: x[m].iloc[0]
                    case 'null':
                        metric_lims[metric] = {'max': np.max(region_results['null_'+metric].apply(np.max)),
                                               'min': np.min(region_results['null_'+metric].apply(np.min))}
                        get_vals = lambda x, m: x['null_'+m].iloc[0]
                    case 'diff':
                        metric_lims[metric] = {'max': np.max(region_results.apply(lambda r: np.max(r[metric] - r['null_'+metric]), axis=1)),
                                               'min': np.min(region_results.apply(lambda r: np.min(r[metric] - r['null_'+metric]), axis=1))}
                        get_vals = lambda x, m: x[m].iloc[0] - x['null_'+m].iloc[0]
                    case 'p_vals':
                        metric_lims[metric] = {'max': np.max(region_results['p_'+metric].apply(np.max)),
                                               'min': np.min(region_results['p_'+metric].apply(np.min))}
                        get_vals = lambda x, m: x['p_'+m].iloc[0]
            
            farc_results = region_results[region_results['kernel'] == 'FARC']
            
            for i, kern in enumerate(kern_types):
                kern_results = region_results[region_results['kernel'] == kern]
    
                for j, (metric, label) in enumerate(zip(metric_cols, metric_labels)):
                    
                    plot_vals = get_vals(kern_results, metric)
                        
                    x_vals = kern_results['ephys_offsets'].iloc[0]
                    y_vals = np.array(kern_results['kernel_widths'].iloc[0]) / 4
                    
                    if kern == 'FARC':
                        ax = axs[i,j]
                        ax.plot(x_vals, plot_vals.flatten())
                        ax.set_title('{}, FARC Kernel'.format(label))
                        ax.set_ylabel(label)
                        ax.set_xlabel('Ephys Lag (s)')
                        
                    else:
                        farc_vals = get_vals(farc_results, metric)
                        
                        x_buff = (x_vals[1] - x_vals[0])/2
                        y_buff = (y_vals[1] - y_vals[0])/2
                        
                        # plot kernel results
                        ax = axs[(i-1)*2+1,j]
                        
                        im = ax.imshow(plot_vals, interpolation=None, aspect='auto', cmap=cmap,
                                       vmax = metric_lims[metric]['max'], vmin = metric_lims[metric]['min'],
                                       extent = (x_vals[0]-x_buff, x_vals[-1]+x_buff, y_vals[-1]+y_buff, y_vals[0]-y_buff))
                        
                        fig.colorbar(im, ax=ax, location='right')
                        
                        ax.set_title('{}, {} Kernel'.format(label, kern))
                        ax.set_ylabel('Kernel Std Dev (s)')
                        ax.set_xlabel('Ephys Lag (s)')
                        
                        # plot difference between FARC and other kernel results
                        ax = axs[(i-1)*2+2,j]

                        im = ax.imshow(farc_vals - plot_vals, interpolation=None, aspect='auto', cmap=cmap,
                                       extent = (x_vals[0]-x_buff, x_vals[-1]+x_buff, y_vals[-1]+y_buff, y_vals[0]-y_buff))
                        
                        fig.colorbar(im, ax=ax, location='right')
                        
                        ax.set_title('{} Diffs (FARC - {} Kernel)'.format(label, kern))
                        ax.set_ylabel('Kernel Std Dev (s)')
                        ax.set_xlabel('Ephys Lag (s)')
                    
                        
# %% Decode time from trial start and end

bin_width = 0.02
fr_kernel = ephys_utils.get_filter_kernel(filter_type='half_gauss', width=0.4, bin_width=bin_width)
    
n_trial_bins = 1 # number of bins to split the trials from each session into for decoding
cpoke_dur_step = 0.3 # width of bins to split the poke durations into for decoding

pre_move_buffer = 0.1
trial_start_buffer = 0.1

ephys_offsets = np.arange(-1, 1, 0.05)
regions = ['FOF', 'ADS', 'PPC']

all_fit_results = []

n_splits = 20
#cv = skl_ms.GroupKFold(n_splits)
cv = skl_ms.ShuffleSplit(n_splits, test_size=0.1)
# perform linear regression of firing rates on click rates using all units
regr = linear_model.LinearRegression()
scores = {'r2': make_scorer(r2_score), 'rmse': make_scorer(root_mean_squared_error)}

def fit_model(x, y, groups):
    if n_splits > 1:
        # catch warnings if split method doesn't use groups
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cv_scores = skl_ms.cross_validate(regr, x, y, groups=groups, cv=cv, scoring=scores)
            results = {'r2': cv_scores['test_r2'], 'rmse': cv_scores['test_rmse']}
    else:
        regr.fit(x, y)
        pred_y = regr.predict(x)
        results = {'r2': r2_score(y, pred_y), 'rmse': root_mean_squared_error(y, pred_y)}
    return results    

for sess_id in [47622, 48261, 67430]: #utils.flatten(sess_ids): #[47622, 48261]: #, 49702, 49636]: # [48261]: # 
    
    all_start = time.perf_counter()
    print('Starting session {}'.format(sess_id))

    sess_trials = sess_data[sess_data['sessid'] == sess_id]
    resp_sel = ((sess_trials['hit'] == 1) | ((sess_trials['FA'] == 1) & (sess_trials['cpoke_dur'] > 0.5))) & sess_trials['valid']
    resp_trials = sess_trials[resp_sel]
    
    # end the stimulus period at cpoke out or stimulus end, whichever comes first
    stim_start = sess_trials['stim_start'].to_numpy()
    cpoke_out = sess_trials['cpoke_out'].to_numpy() - pre_move_buffer
    
    rel_cpoke_out = cpoke_out - stim_start
    rel_poke_bounds = np.hstack((np.full_like(rel_cpoke_out, trial_start_buffer)[:,None], rel_cpoke_out[:,None]))
    resp_poke_bounds = rel_poke_bounds[resp_sel,:]
    
    # get firing rates for the stimulus period filter the units whose number of spikes is at least 1 Hz for nosepoke periods
    sess_units = unit_data[unit_data['sessid'] == sess_id]
    trial_start_ts = sess_units['trial_start_timestamps'].iloc[0][0:-1]

    abs_stim_start = trial_start_ts + stim_start
    abs_cpoke_out = trial_start_ts + cpoke_out

    poke_frs = ephys_utils.get_avg_period_fr(sess_units['spike_timestamps'], np.hstack((abs_stim_start[:,None], abs_cpoke_out[:,None])))
    
    sess_units = sess_units[poke_frs > 1]

    sess_regions = np.intersect1d(np.unique(sess_units['region']), regions)

    for region in sess_regions:
        region_start = time.perf_counter()
        region_units = sess_units[sess_units['region'] == region]
        
        fit_results = []
        
        for i, ephys_offset in enumerate(ephys_offsets):
            
            offset_start = time.perf_counter()
            
            # compute the smoothed firing rate matrices by trial
            fr_mats, ts = ephys_utils.get_fr_matrix_by_trial(region_units['spike_timestamps'], abs_stim_start + ephys_offset, fr_kernel, rel_poke_bounds, resp_sel)
            ts = ts.apply(lambda t: t[:,None])
            
            poke_dur_bin_edges = np.arange(trial_start_buffer, np.max(resp_trials['cpoke_dur'])-pre_move_buffer+cpoke_dur_step, cpoke_dur_step)
    
            # get all trials together as well as split into bins for comparison
            if n_trial_bins > 1:
                trial_bins_to_use = [1, n_trial_bins]
            else:
                trial_bins_to_use = [1]
                
            for trial_bins in trial_bins_to_use:
                trial_bin_edges = np.quantile(resp_trials['trial'], np.linspace(0,1,trial_bins+1))
                
                for i in range(trial_bins):
                    trial_sel = (resp_trials['trial'] >= trial_bin_edges[i]) & (resp_trials['trial'] <= trial_bin_edges[i+1])
                    trial_sel = trial_sel.to_numpy()
                    if trial_bins == 1:
                        trials_label = 'All'
                    else:
                        trials_label = '{:.0f}-{:.0f}'.format(trial_bin_edges[i], trial_bin_edges[i+1])
                    
                    trial_frs = fr_mats[trial_sel]
                    trial_poke_bounds = resp_poke_bounds[trial_sel,:]

                    # look at the whole trial as well
                    for whole_trial in [True, False]:
                        # Have time be from either stimulus start or stimulus end
                        for forward in [True, False]:
                            trial_ts = ts[trial_sel]
                            if forward:
                                alignment = 'Cpoke In'
                            else:
                                trial_ts = trial_ts.apply(lambda t: np.arange(-len(t), 0, 1)[:,None]*bin_width)
                                alignment = 'Cpoke Out'
                                
                            if whole_trial:
                                dur_label = 'All'
                                
                                stacked_frs = np.vstack(trial_frs)
                                stacked_ts = np.vstack(trial_ts)
                                trial_nums = np.vstack([np.full((len(t), 1), i) for i, t in enumerate(trial_ts)])
                                
                                fit_scores = fit_model(stacked_frs, stacked_ts, trial_nums)
        
                                fit_results.append({'trials': trials_label,
                                                    'time': dur_label,
                                                    'alignment': alignment,
                                                    'n_trials': np.sum(trial_sel),
                                                    'ephys_offset': ephys_offset,
                                                    'r2': np.mean(fit_scores['r2']),
                                                    'rmse': np.mean(fit_scores['rmse'])})

                            else:
                                
                                for j in range(len(poke_dur_bin_edges)-1):
                                    dur_bin_start_idx = int((poke_dur_bin_edges[j]-trial_start_buffer)/bin_width)
                                    dur_bin_end_idx = int((poke_dur_bin_edges[j+1]-trial_start_buffer)/bin_width)
                                    dur_trial_sel = resp_trials[trial_sel]['cpoke_dur']-pre_move_buffer >= poke_dur_bin_edges[j]
                                    dur_trial_sel = dur_trial_sel.to_numpy()
                                    dur_label = '{:.1f}-{:.1f}'.format(poke_dur_bin_edges[j], poke_dur_bin_edges[j+1])
                                    
                                    if np.sum(dur_trial_sel) < n_splits:
                                        continue
                                    
                                    if forward:
                                        sub_ts = [t[dur_bin_start_idx:dur_bin_end_idx] if len(t) > dur_bin_end_idx 
                                                   else t[dur_bin_start_idx:] for t in trial_ts[dur_trial_sel]]
                                        sub_frs = [m[dur_bin_start_idx:dur_bin_end_idx,:] if m.shape[0] > dur_bin_end_idx 
                                                    else m[dur_bin_start_idx:,:] for m in trial_frs[dur_trial_sel]]
                                    else:
                                        sub_ts = [np.flip(t)[dur_bin_start_idx:dur_bin_end_idx] if len(t) > dur_bin_end_idx 
                                                   else np.flip(t)[dur_bin_start_idx:] for t in trial_ts[dur_trial_sel]]
                                        sub_frs = [np.flip(m, axis=0)[dur_bin_start_idx:dur_bin_end_idx,:] if m.shape[0] > dur_bin_end_idx 
                                                    else np.flip(m, axis=0)[dur_bin_start_idx:,:] for m in trial_frs[dur_trial_sel]]
                                        
                                    stacked_frs = np.vstack(sub_frs)
                                    stacked_ts = np.vstack(sub_ts)
                                    trial_nums = np.vstack([np.full((len(t), 1), i) for i, t in enumerate(sub_ts)])

                                    fit_scores = fit_model(stacked_frs, stacked_ts, trial_nums)
            
                                    fit_results.append({'trials': trials_label,
                                                        'time': dur_label,
                                                        'alignment': alignment,
                                                        'n_trials': np.sum(trial_sel),
                                                        'ephys_offset': ephys_offset,
                                                        'r2': np.mean(fit_scores['r2']),
                                                        'rmse': np.mean(fit_scores['rmse'])})  
            
            print('    Processed ephys offset {:.2f} in {:.1f} s'.format(ephys_offset, time.perf_counter()-offset_start))
                                    
        fit_results = pd.DataFrame(fit_results)
        
        all_fit_results.append({'subjid': sess_trials['subjid'].iloc[0],
                                'sessid': sess_id,
                                'region': region,
                                'n_units': len(region_units),
                                'results': fit_results})
        
        print('  Processed region {} in {:.1f} s'.format(region, time.perf_counter()-region_start))
   
    print('Processed session {} in {:.1f} s'.format(sess_id, time.perf_counter()-all_start))
    
save_path = path.join(utils.get_user_home(), 'db_data', 'trial_time_regression.pkl')

with open(save_path, 'wb') as f:
    pickle.dump(all_fit_results, f)

# %%

cmap = 'jet'

for fit_result_data in all_fit_results:
    fit_results = fit_result_data['results']

    trial_groups = np.unique(fit_results['trials'])
    if any(trial_groups != 'All'):
        trial_groups = sorted(trial_groups[trial_groups != 'All'], key=lambda x: int(x.split('-')[0]))
        trial_groups = np.insert(trial_groups, 0, 'All')
    n_groups = len(trial_groups)
    
    row_heights = [1]+[3 for i in range(n_groups)]

    for plot_col, col_label in zip(['r2', 'rmse'], ['R2', 'RMSE']): #[['rmse', 'RMSE']]: #

        fig, axs = plt.subplots(len(row_heights), 2, layout='constrained', figsize=(10, np.sum(row_heights)), height_ratios=row_heights, sharey='row')
        fig.suptitle('Time Decoding. Subject {}, Session {}, {} (n={}): {}'.format(fit_result_data['subjid'], fit_result_data['sessid'], 
                                                                    fit_result_data['region'], fit_result_data['n_units'], col_label))
    
        all_time_trials = fit_results[(fit_results['time'] == 'All')]
        time_bin_trials = fit_results[(fit_results['time'] != 'All')]
        
        for i, forward in enumerate([True, False]):
            if forward:
                alignment = 'Cpoke In'
            else:
                alignment = 'Cpoke Out'
            
            # handle all time results first
            all_time_alignment = all_time_trials[all_time_trials['alignment'] == alignment]
            ax = axs[0,i]
            for group in trial_groups:
                plot_vals = all_time_alignment.loc[all_time_alignment['trials'] == group, ['ephys_offset', plot_col]]
                ax.plot(plot_vals['ephys_offset'], plot_vals[plot_col], label=group)
            ax.set_xlabel('Ephys Lag (s)')
            ax.set_ylabel(col_label)
            ax.set_title('All Time Aligned to {}'.format(alignment))
            ax.legend()
            ax.yaxis.set_tick_params(which='both', labelleft=True)

            # if plot_col == 'r2':
            #     ax.set_ylim(0, 1)
            
            # handle time bins for each group
            time_bin_alignment = time_bin_trials[time_bin_trials['alignment'] == alignment]
            for j, group in enumerate(trial_groups):
                group_time_bin_alignment = time_bin_alignment[time_bin_alignment['trials'] == group]
                
                # build value matrix
                ephys_offsets = np.unique(group_time_bin_alignment['ephys_offset'])
                time_bins = sorted(np.unique(group_time_bin_alignment['time']), key=lambda x: float(x.split('-')[0]))
                
                vals = np.zeros((len(time_bins), len(ephys_offsets)))
                for ti, time_bin in enumerate(time_bins):
                    for oi, offset in enumerate(ephys_offsets):
                        vals[ti, oi] = group_time_bin_alignment.loc[
                            (group_time_bin_alignment['ephys_offset'] == offset) & (group_time_bin_alignment['time'] == time_bin), plot_col].iloc[0]
                        
                x_buff = (ephys_offsets[1] - ephys_offsets[0])/2
                
                # plot values
                ax = axs[j+1,i]
                
                im = ax.imshow(vals, interpolation=None, aspect='auto', cmap=cmap,
                               vmin = np.quantile(vals, 0.05), vmax = np.quantile(vals, 0.95),
                               extent = (ephys_offsets[0]-x_buff, ephys_offsets[-1]+x_buff, len(time_bins)-0.5, -0.5))
                
                fig.colorbar(im, ax=ax, location='right')
                
                ax.set_title('Time Groups for Trial Group {} aligned to {}'.format(group, alignment))
                ax.set_ylabel('Time Bin')
                ax.set_xlabel('Ephys Lag (s)')
                ax.set_yticks(np.arange(len(time_bins)))
                ax.set_yticklabels(time_bins)
                ax.yaxis.set_tick_params(which='both', labelleft=True)
    
                
                