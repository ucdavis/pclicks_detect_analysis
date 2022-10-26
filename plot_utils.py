# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 15:41:12 2022

@author: tanne
"""

import matplotlib.pyplot as plt
import utils


def plot_psth(signal, time, error=None, ax=None, plot_x0=True, **kwargs):

    # if no axes are passed in, create an axis
    if ax is None:
        # if there are no figures, create one first
        if len(plt.get_fignums()) == 0:
            fig, ax = plt.subplots(1, 1)
        else:
            ax = plt.gca()

    # plot line at x=0
    if plot_x0:
        ax.axvline(dashes=[4, 4], c='k', lw=1)

    if not error is None:
        # plot error first
        upper = signal + error
        lower = signal - error
        fill = ax.fill_between(time, upper, lower, alpha=0.2, **kwargs)
        # make sure the fill is the same color as the signal line
        c = fill.get_facecolor()
        ax.plot(time, signal, color=c[:, 0:3], **kwargs)
    else:
        # just plot signal
        ax.plot(time, signal, **kwargs)

    return ax


def plot_psth_dict(psth_dict, ax=None, plot_x0=True, **kwargs):

    return plot_psth(psth_dict['signal_avg'], psth_dict['time'], psth_dict['signal_se'], ax, plot_x0, **kwargs)


def plot_raster(spike_times, ax=None, plot_x0=True, **kwargs):

    # if no axes are passed in, create an axis
    if ax is None:
        # if there are no figures, create one first
        if len(plt.get_fignums()) == 0:
            fig, ax = plt.subplots(1, 1)
        else:
            ax = plt.gca()

    # plot line at x=0
    if plot_x0:
        ax.axvline(dashes=[4, 4], c='k', lw=1)

    # determine if there are multiple trials to stack or not
    if len(spike_times) > 0 and utils.is_scalar(spike_times[0]):
        spike_times = [spike_times]

    # iterate through trials and plot the spikes stacked one on top of the other
    for i, trial_spike_times in enumerate(spike_times):
        y_min = [i] * len(trial_spike_times)
        y_max = [i+1] * len(trial_spike_times)
        ax.vlines(trial_spike_times, y_min, y_max)

    return ax
