import os
import sys
import os


import utils.variables as v
from utils.data import read_eeg_data, read_psd_data

def generate_all_recs():
    """
    Generate all possible recording names based on the number of participants, sessions, and runs.
    Returns
    -------
    recs : list of str
        List of all possible recording names in the format 'P00X_S00X_00X'.
    """
    print("---- Generating all recordings ----")
    recs = []
    for i in range(1, v.NUM_SUBJECTS + 1):
        subject = f'P{str(i).zfill(3)}'
        for j in range(1, v.NUM_SESSIONS + 1):
            session = f'S{str(j).zfill(3)}'
            for k in range(1, v.NUM_RUNS + 1):
                run = f'{str(k).zfill(3)}'
                rec = f'{subject}_{session}_{run}'
                recs.append(rec)
    print("All records generated")
    return recs

def filter_valid_recs(recs, data_type, output_type):
    """
    This function returns the valid recordings from a list of recordings.
    Parameters
    ----------
    recs : list
        A list of recording names in the format 'P00X_S00X_00X'.
    Returns
    -------
    valid_recs : list
        A list of valid recordings in the format 'P00X_S00X_00X'.
    """
    if data_type == 'raw':
        dir = v.DIR_RAW
    elif data_type == 'ica':
        dir = v.DIR_ICA_FILTERED
    elif data_type == 'init':
        dir = v.DIR_INIT_FILTERED
    elif data_type == 'new_ica':
        dir = v.DIR_NEW_ICA
    elif data_type == 'psd':
        dir = v.DIR_PSD
    else:
        print(f'No data with data_type = {data_type} found')
        return 0

    print('\n---- Filtering out invalid recordings ----')
    valid_recs = []
    
    for rec in recs:
        subject, session, run = rec.split('_')
        f_name = os.path.join(dir, f'sub-{subject}_ses-{session}_run-{run}.mat')
        if data_type == 'psd':
            try:
                data = read_psd_data(f_name)
                if data.shape[1] >= v.NUM_PSD_FREQS:
                    valid_recs.append(rec)
                else:
                    print(f'{f_name} not valid file name')
            except:
                print(f"ERROR 1) Failed to read data for recording {rec}")
                continue
        try:
            data = read_eeg_data(data_type, f_name, output_type)
            if output_type == 'mne' and data.n_times / data.info['sfreq'] >= 5 * 60:
                valid_recs.append(rec)
            elif output_type == 'np':
                if (data_type=='ica' or data_type=='raw' or data_type=='init') and data.shape[1] >= v.NUM_SAMPLES:
                    valid_recs.append(rec)
                if data_type == 'new_ica' and data.shape[1]>= v.NEW_NUM_SAMPLES:
                    valid_recs.append(rec)
            else:
                print(f'{f_name} not valid file name')
        except:
            print(f"ERROR 1) Failed to read data for recording {rec}")
            continue
    print('\n---- Returning valid recordings ----')
    print(valid_recs)
    return valid_recs


def get_valid_recs(data_type, output_type):
    """
    This function returns a list of valid recording names based on the raw EEG data.
    Returns
    -------
    valid_recs : list
        A list of valid recording names in the format 'P00X_S00X_00X'.
    """

    recs = generate_all_recs()
    valid_recs = filter_valid_recs(recs, data_type, output_type)
    return valid_recs
