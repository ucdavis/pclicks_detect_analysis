# -*- coding: utf-8 -*-
"""
local database for poisson clicks detect protocol

@author: tanner stevenson
"""

from . import base_db
import numpy as np
import utils


class LocalDB_PClicksDetect(base_db.LocalDB_Base):

    def __init__(self, save_locally=True, reload=False, data_dir=None):
        super().__init__(save_locally, reload, data_dir)

    @property
    def protocol_name(self):
        ''' PClicksDetect '''
        return 'PClicksDetect'

    def _format_sess_data(self, sess_data):
        ''' Format the session data based on the PClicksDetect protocol to extract relevant timepoints '''

        # separate parsed event history into states and events dictionaries for use later
        peh = sess_data['parsed_events'].transform({'states': lambda x: x['States'], 'events': lambda x: x['Events']})

        # simplify some column names
        sess_data.rename(columns={'choice': 'outcome', 'viol': 'violation',
                                  'click_times': 'rel_click_times'}, inplace=True)

        # add some additional columns to filter on
        sess_data['correct_trial'] = (sess_data['hit'] == 1) | (sess_data['CR'] == 1)
        sess_data['rewarded'] = peh['states'].apply(lambda x: not x['Reward'][0] is None)

        # determine trial alignment points
        # when the stimulus starts being played
        sess_data['stim_start'] = peh['events'].apply(lambda x: np.nan if not 'BNC1High' in x else x['BNC1High'])

        # click times from trial start
        sess_data['abs_click_times'] = sess_data['rel_click_times'] + sess_data['stim_start']

        # change time from trial start
        sess_data['change_time'] = sess_data['stim_start'] + sess_data['change_delay']
        # false alarms, catch trials, and trials with no outcome don't have a change time
        sess_data.loc[(sess_data['catch_trial'] == 1) | (sess_data['FA'] == 1) |
                      (sess_data['outcome'] == 'none'), 'change_time'] = np.nan

        # pokes in and out
        sess_data['cpoke_start'] = peh['states'].apply(lambda x: x['WaitForCenterPoke'][1])
        sess_data = sess_data.apply(self.__find_stim_end_poke_out, axis=1)

        # reaction time
        sess_data['RT'] = sess_data['cpoke_out'] - sess_data['change_time']
        # misses don't have a reaction time
        sess_data.loc[sess_data['miss'] == 1, 'RT'] = np.nan

        # reward poke
        sess_data['reward_time'] = peh['states'].apply(lambda x: np.nan if x['Reward'][0] is None else x['Reward'][0])

        # trial end
        sess_data['trial_end'] = peh['states'].apply(lambda x: x['ITI'][1])

        # calculate some metrics about the trials
        sync_sent = peh['states'].apply(lambda x: np.nan if x['StimulusOne'][0] is None else min(x['StimulusOne']))
        sess_data['sync_lag'] = sess_data['stim_start'] - sync_sent
        sess_data = sess_data.apply(self.__determine_valid, axis=1)

        return sess_data

    def __find_stim_end_poke_out(self, row):
        # use the behavioral states and events to reconstruct when the stimulus ended
        # and the animal removed its nose from the center port
        outcome = row['outcome']
        states = row['parsed_events']['States']
        events = row['parsed_events']['Events']
        # hits
        if outcome == 'rewarded hit' or outcome == 'unrewarded hit':
            stim_end = states['Hit'][0]
            cpoke_out = stim_end
        # misses
        elif outcome == 'miss':
            stim_end = states['Miss'][0]
            if 'Port2Out' in events:
                # find when the animal first removed their nose after a miss
                out_after_miss = (np.array(events['Port2Out']) > states['Miss'][1])
                if np.any(out_after_miss):
                    if utils.is_scalar(out_after_miss):
                        cpoke_out = events['Port2Out']
                    else:
                        cpoke_out = events['Port2Out'][np.flatnonzero(out_after_miss)[0]]
                else:
                    cpoke_out = np.nan
            else:
                cpoke_out = np.nan
        # false alarms
        elif outcome == 'false alarm':
            if states['LegalCenterBreakTimer'][0] is None:
                # catch trials will go straight to false alarm
                cpoke_out = states['FalseAlarm'][0]
            else:
                # get the second to last timestamp indicating the start of the final legal center break
                cpoke_out = sorted(states['LegalCenterBreakTimer'])[-2]

            stim_end = states['FalseAlarm'][0]
        # correct rejections
        elif outcome == 'rewarded CR' or outcome == 'unrewarded CR':
            stim_end = states['WaitForCRCenterOut'][0]
            cpoke_out = states['WaitForCRCenterOut'][1]
        else:
            stim_end = np.nan
            cpoke_out = np.nan

        row['stim_end'] = stim_end
        row['cpoke_out'] = cpoke_out
        return row

    def __determine_valid(self, row):
        # checks the timing on trials to find trials where a bug in the state machine caused incorrect state progression
        outcome = row['outcome']
        change_time = row['stim_start'] + row['change_delay']
        rw_end = change_time + row['response_window']
        if outcome == 'rewarded hit':
            valid = row['cpoke_out'] > change_time and row['cpoke_out'] < rw_end
        elif outcome == 'miss':
            valid = (row['stim_end'] - rw_end) > -1e-3
        elif outcome == 'false alarm':
            valid = ((row['cpoke_out'] < change_time and row['catch_trial'] == 0) or
                     (row['cpoke_out'] < rw_end and row['catch_trial'] == 1))
        elif outcome == 'rewarded CR':
            valid = (row['stim_end'] - rw_end) > -1e-3
        else:
            valid = False

        row['valid'] = valid
        return row
