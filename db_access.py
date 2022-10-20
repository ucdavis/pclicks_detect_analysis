# -*- coding: utf-8 -*-
"""
Data access layer for hanks lab database

@author: tanner stevenson
"""

import mysql.connector
import os.path as path
import pandas as pd
import numpy as np
import numbers
import json
import time
import utils
import math

## Unit and Session Data ##


def get_unit_data(unit_ids):
    '''Gets all ephys data for the given unit ids'''

    if utils.is_scalar(unit_ids):
        unit_ids = [unit_ids]

    print('Retrieving {0} units...'.format(len(unit_ids)))
    start = time.perf_counter()

    db = __get_connector()
    cur = db.cursor(dictionary=True, buffered=True)

    query = 'select * from met.units where unitid in ({0})'

    max_rows = 500  # number of rows to retrieve at once

    if len(unit_ids) < max_rows:
        cur.execute(query.format(','.join([str(i) for i in unit_ids])))
        db_data = cur.fetchall()
    else:
        n_iter = math.ceil(len(unit_ids)/max_rows)
        batch_start = time.perf_counter()
        for i in range(n_iter):

            # get batch of unit ids to load
            if i < n_iter:
                batch_ids = unit_ids[i*max_rows:(i+1)*max_rows]
            else:
                batch_ids = unit_ids[i*max_rows:]

            # load data
            cur.execute(query.format(','.join([str(i) for i in batch_ids])))
            rows = cur.fetchall()

            if i == 0:
                db_data = rows
            else:
                db_data = db_data + rows

            print('Retrieved {0}/{1} units in {2:.1f} s'.format(i*max_rows+cur.rowcount,
                  len(unit_ids), time.perf_counter()-batch_start))

    # read out data stored in json
    for i, row in enumerate(db_data):
        db_data[i]['spike_timestamps'] = np.array(__parse_json(row['spike_timestamps']))
        db_data[i]['trial_start_timestamps'] = np.array(__parse_json(row['trial_start_timestamps']))
        db_data[i]['waveform'] = __parse_json(row['waveform'])

    # convert to data table
    unit_data = pd.DataFrame.from_dict(db_data)

    db.close()

    print('Retrieved {0} units in {1:.1f} s'.format(len(unit_ids), time.perf_counter()-start))

    return unit_data.sort_values('unitid')


def get_session_data(session_ids):
    '''Gets all behavioral data for the given session ids'''

    if utils.is_scalar(session_ids):
        session_ids = [session_ids]

    if len(session_ids) > 1:
        print('Retrieving {0} sessions...'.format(len(session_ids)))

    start = time.perf_counter()

    db = __get_connector()
    cur_sess = db.cursor(dictionary=True, buffered=True)
    cur_trial = db.cursor(dictionary=True, buffered=True)

    id_str = ','.join([str(i) for i in session_ids])

    sess_query = ('select sessid, subjid, sessiondate, starttime, protocol, startstage, rigid '
                  'from beh.sessions where sessid in ({0}) order by sessid').format(id_str)

    trial_query = ('select sessid, trialnum, data, parsed_events from beh.trials '
                   'where sessid in ({0}) order by sessid, trialnum')

    # get all session data
    cur_sess.execute(sess_query)
    sess_rows = cur_sess.fetchall()

    sess_data = []

    sess_start = time.perf_counter()
    for i, sess in enumerate(sess_rows):

        # fetch all trials for this session
        cur_trial.execute(trial_query.format(sess['sessid']))
        trials = cur_trial.fetchall()

        for trial in trials:
            # read out data stored in json
            trial['parsed_events'] = __parse_json(trial['parsed_events'])
            # remove data into its own dictionary
            trial_data = __parse_json(trial.pop('data'))
            trial_data.pop('n_done_trials')  # this is redundant
            # convert all lists of numbers to numpy arrays
            for key, value in trial_data.items():
                if not utils.is_scalar(value) and isinstance(value[0], numbers.Number):
                    trial_data[key] = np.array(value)
            # merge all dictionaries into single row
            sess_data.append({**sess, **trial, **trial_data})

        if (i % 5 == 0 or i == len(sess_rows)-1) and not i == 0:
            print('Retrieved {0}/{1} sessions in {2:.1f} s'.format(i +
                  1, len(session_ids), time.perf_counter()-sess_start))

    sess_data = pd.DataFrame.from_dict(sess_data)
    sess_data.rename(columns={'trialnum': 'trial'}, inplace=True)

    db.close()

    print('Retrieved {0} sessions in {1:.1f} s'.format(len(session_ids), time.perf_counter()-start))

    return sess_data.sort_values(['sessid', 'trial'])


## Unit and Session IDs ##

def get_unit_protocol_subj_ids(protocol):
    ''' Gets all subject ids with unit information for a particular protocol '''
    db = __get_connector()
    cur = db.cursor(buffered=True)

    cur.execute('select distinct a.subjid from beh.sessions a, met.units b where protocol=\'{0}\' and a.sessid=b.sessid'
                .format(protocol))
    ids = cur.fetchall()

    # flatten list of tuples
    return [i[0] for i in ids]


def get_subj_unit_ids(subj_ids):
    '''Gets all unit ids for the given subject ids. Returns a dictionary of unit ids indexed by subject id'''

    if utils.is_scalar(subj_ids):
        subj_ids = [subj_ids]

    db = __get_connector()
    cur = db.cursor(buffered=True, dictionary=True)

    cur.execute('select subjid, unitid from met.units where subjid in ({0})'
                .format(','.join([str(i) for i in subj_ids])))
    ids = cur.fetchall()

    # group the unit ids by subject
    # Note: this is much faster than repeatedly querying the database
    df = pd.DataFrame.from_dict(ids)
    # group unit ids into a sorted list by subject id
    df = df.groupby('subjid').agg(list)['unitid'].apply(lambda x: sorted(x))

    return df.to_dict()


def get_sess_unit_ids(sess_ids):
    '''Gets all unit ids for the given session ids. 
    Returns a dictionary of unit ids indexed by session id'''

    if utils.is_scalar(sess_ids):
        sess_ids = [sess_ids]

    db = __get_connector()
    cur = db.cursor(buffered=True, dictionary=True)

    cur.execute('select sessid, unitid from met.units where sessid in ({0})'
                .format(','.join([str(i) for i in sess_ids])))
    ids = cur.fetchall()

    # group the unit ids by session
    # Note: this is much faster than repeatedly querying the database
    df = pd.DataFrame.from_dict(ids)
    # group unit ids into a sorted list by session id
    df = df.groupby('sessid').agg(list)['unitid'].apply(lambda x: sorted(x))

    return df.to_dict()


def get_subj_unit_sess_ids(subj_ids):
    '''Gets all session ids that have unit data for the given subject ids. 
    Returns a dictionary of session ids indexed by subject id'''

    if utils.is_scalar(subj_ids):
        subj_ids = [subj_ids]

    db = __get_connector()
    cur = db.cursor(buffered=True, dictionary=True)

    cur.execute('select distinct sessid, subjid from met.units where subjid in ({0})'
                .format(','.join([str(i) for i in subj_ids])))
    ids = cur.fetchall()

    # group the session ids by subject
    # Note: this is much faster than repeatedly querying the database
    df = pd.DataFrame.from_dict(ids)
    # group session ids into a sorted list by subject id
    df = df.groupby('subjid').agg(list)['sessid'].apply(lambda x: sorted(x))

    return df.to_dict()


def get_unit_sess_ids(unit_ids):
    '''Gets all session ids for the given unit ids. 
    Returns a dictionary of unit ids indexed by session id'''

    if utils.is_scalar(unit_ids):
        unit_ids = [unit_ids]

    db = __get_connector()
    cur = db.cursor(buffered=True, dictionary=True)

    cur.execute('select sessid, unitid from met.units where unitid in ({0})'
                .format(','.join([str(i) for i in unit_ids])))
    ids = cur.fetchall()

    # group the unit ids by session
    # Note: this is much faster than repeatedly querying the database
    df = pd.DataFrame.from_dict(ids)
    # group unit ids into a sorted list by session id
    df = df.groupby('sessid').agg(list)['unitid'].apply(lambda x: sorted(x))

    return df.to_dict()


## PRIVATE METHODS ##


def __get_connector():
    '''Private method to get the database connection'''

    config_path = path.join(path.expanduser('~'), '.dbconf')
    conn_info = {}

    if path.exists(config_path):
        # read db connection information from file
        with open(config_path) as f:
            for line in f:
                if '=' in line:
                    prop_val = line.split('=')
                    conn_info[prop_val[0].strip()] = prop_val[1].strip()
    else:
        # create a new db config
        # get values from user
        val = input('Enter host address: ')
        conn_info['host'] = val
        val = input('Enter username: ')
        conn_info['user'] = val
        val = input('Enter password: ')
        conn_info['passwd'] = val

        # write file
        with open(config_path, 'w') as config:
            config.write('[client]\n')
            for name, val in conn_info.items():
                config.write('{0} = {1}\n'.format(name, val))

    try:
        con = mysql.connector.connect(
            host=conn_info['host'],
            user=conn_info['user'],
            password=conn_info['passwd'])

        return con
    except BaseException as e:
        print('Could not connect to the database: {0}'.format(e.msg))
        raise


def __parse_json(x):
    '''Private method to convert json to values'''
    return json.loads(x.decode('utf-8'))['vals']
