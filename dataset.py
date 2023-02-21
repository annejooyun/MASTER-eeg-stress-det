import scipy
import os
import numpy as np
import pandas as pd
import variables as v
import logging
import mne
from sklearn.model_selection import StratifiedKFold

def read_eeg_data(data_type, filename):
    print('im in')
    if data_type == 'raw':
        data_key = 'raw_eeg_data'
    elif data_type == 'ica':
        data_key = 'Clean_data'
    else:
        print(f'No data with data_type = {data_type} found')
        return 0
    print(filename)
    data = scipy.io.loadmat(filename)[data_key]
    info = mne.create_info(8, sfreq=v.SFREQ, ch_types= 'eeg', verbose=None)
    raw_arr = mne.io.RawArray(data, info)
    mapping = {'0':'F4','1':'Fp2','2':'C3','3':'FC6','4':'O1','5':'Oz','6':'FT9','7':'T8'}
    mne.rename_channels(raw_arr.info, mapping)

    return raw_arr

def generate_all_recs():
    """
    Generate all possible recording names based on the number of participants, sessions, and runs.
    Returns
    -------
    recs : list of str
        List of all possible recording names in the format 'P00X_S00X_00X'.
    """

    recs = []
    for i in range(1, v.NUM_SUBJECTS + 1):
        subject = f'P{str(i).zfill(3)}'
        for j in range(1, v.NUM_SESSIONS + 1):
            session = f'S{str(j).zfill(3)}'
            for k in range(1, v.NUM_RUNS + 1):
                run = f'{str(k).zfill(3)}'
                rec = f'{subject}_{session}_{run}'
                recs.append(rec)
    return recs


def filter_valid_recs(recs, data_type = 'ica'):
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
    else:
        print(f'No data with data_type = {data_type} found')
        return 0

    valid_recs = []
    for rec in recs:
        subject, session, run = rec.split('_')
        f_name = os.path.join(
            dir, f'sub-{subject}_ses-{session}_run-{run}.mat')
        try:
            data = read_eeg_data(data_type, f_name)
            if data.n_times / data.info['sfreq'] >= 4.30 * 60:
                valid_recs.append(rec)
        except:
            logging.error(f"1) Failed to read data for recording {rec}")
            continue
    return valid_recs


def get_valid_recs():
    """
    This function returns a list of valid recording names based on the raw EEG data.
    Returns
    -------
    valid_recs : list
        A list of valid recording names in the format 'P00X_S00X_00X'.
    """

    recs = generate_all_recs()
    valid_recs = filter_valid_recs(recs)
    return valid_recs


def extract_eeg_data(valid_recs, data_type="ica"):
    '''
    Loads data from the dataset.
    The data_type parameter specifies which of the datasets to load. Possible values
    are raw and ica_filtered.
    Returns
    '''
    assert (data_type in v.DATA_TYPES)

    if data_type == "raw":
        dir = v.DIR_RAW
        data_key = 'Data'
    elif data_type == "ica":
        dir = v.DIR_ICA_FILTERED
        data_key = 'Clean_data'
    else:
        print("No files matching data type found")
        return 0

    eeg_data = {}
    for rec in valid_recs:
        subject, session, run = rec.split('_')
        f_name = os.path.join(
            dir, f'sub-{subject}_ses-{session}_run-{run}.mat')
        try:
            data = read_eeg_data(data_type, f_name)
        except:
            logging.error(f"2) Failed to read data for recording {rec}")
            data = None
        key = f"{subject}_{session}_{run}"
        eeg_data[key] = data
    return eeg_data


def compute_average_scores(path='Data/STAI_grading.xlsx'):
    """
    Compute the average scores of subjects from the raw scores.
    Parameters
    ----------
    path : str
        Path to the raw scores excel file.
    Returns
    -------
    scores_df : pandas.DataFrame
        DataFrame containing the average scores of subjects
    """

    y1_text = 'Total score Y1'
    y2_text = 'Total score Y2'
    columns = ['SubjectNo', 'D1Y1', 'D2Y1', 'J1Y1', 'J2Y1', 'D1Y2',
               'D2Y2', 'J1Y2', 'J2Y2', 'AVGD1', 'AVGD2', 'AVGJ1', 'AVGJ2']

    try:
        xl = pd.ExcelFile(path)
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
        return

    n_subjects = len(xl.sheet_names)-2
    scores = []

    sheet_names = xl.sheet_names.copy()
    sheet_names.remove('MAL')
    sheet_names.remove('Rating 1-10')

    for sheet_name in sheet_names:
        sheet = xl.parse(sheet_name)
        y1_values = sheet.loc[sheet[sheet.columns[0]] == y1_text].iloc[:, 1:].to_numpy()[
            0][::2]
        y2_values = sheet.loc[sheet[sheet.columns[0]] == y2_text].iloc[:, 1:].to_numpy()[
            0][::2]
        avg_d1 = np.mean(np.hstack((y1_values[0], y2_values[0])))
        avg_d2 = np.mean(np.hstack((y1_values[1], y2_values[1])))
        avg_j1 = np.mean(np.hstack((y1_values[2], y2_values[2])))
        avg_j2 = np.mean(np.hstack((y1_values[3], y2_values[3])))
        scores.append(
            np.hstack((y1_values, y2_values, avg_d1, avg_d2, avg_j1, avg_j2)))
    scores_df = pd.DataFrame(scores, columns=columns[1:])
    scores_df.insert(0, columns[0], range(1, n_subjects+1))
    return scores_df


def compute_score_labels(scores, low_cutoff, high_cutoff):
    """
    Convert scores to labels based on low and high cutoffs.
    Parameters
    ----------
    scores : pd.DataFrame
        Dataframe containing the scores for all subjects.
    low_cutoff : int, optional
        The lower cutoff value. Default is 37.
    high_cutoff : int, optional
        The upper cutoff value. Default is 45.
    Returns
    -------
    numpy.ndarray
        An array with shape `(n_subjects, 4)` containing labels.
    """

    labels = {}
    for i in range(scores.shape[0]):
        for j in range(v.NUM_SESSIONS*v.NUM_RUNS):

            label = int(scores.iloc[i, j+8]
                        in range(low_cutoff, high_cutoff+1))
            if scores.iloc[i, j+8] > high_cutoff:
                label = 2
            subject = i + 1
            session = j//v.NUM_RUNS + 1
            run = j % v.NUM_RUNS + 1
            key = f'P{str(subject).zfill(3)}_S{str(session).zfill(3)}_{str(run).zfill(3)}'
            labels[key] = label
    return labels


def get_labels(valid_recs, path='Data/STAI_grading.xlsx', low_cutoff=37, high_cutoff=45):
    """
    Get labels for valid recordings.
    Parameters
    ----------
    valid_recs : list
        List of valid recordings.
    path : str, optional
        Path to the raw scores excel file. Default is 'Data/STAI_grading.xlsx'.
    low_cutoff : int, optional
        The lower cutoff value. Default is 37.
    high_cutoff : int, optional
        The upper cutoff value. Default is 45.
    Returns
    -------
    dict
        Dictionary containing labels for valid recordings.
    """
    scores = compute_average_scores(path)
    labels = compute_score_labels(scores, low_cutoff, high_cutoff)
    return labels



def extract_epochs(x_dict, y_dict, epoch_duration=3):
    """
    Extracts epochs of given duration from each sample in x_dict.
    Parameters
    ----------
    x_dict : dict
        Dictionary of MNE raw objects, where each object is a sample.
    y_dict : dict
        Dictionary of labels for each sample in x_dict.
    epoch_duration : float, optional
        Duration of each epoch, in seconds, by default 3.
    Returns
    -------
    x_epochs : dict
        A dictionary of MNE Epoch objects, where each sample in x_epochs is split into epochs of the given duration.
    y_epochs : dict
        A dictionary of labels for each epoch in x_epochs.
    """

    overlap_duration = 0.0  # in seconds

    x_epochs = {}
    y_epochs = {}

    for key, raw in x_dict.items():
        events = mne.make_fixed_length_events(
            raw, duration=epoch_duration, overlap=overlap_duration)
        epochs = mne.Epochs(raw, events, tmin=0,
                            tmax=epoch_duration, baseline=None, preload=True)

        for i, epoch in enumerate(epochs):
            x_epochs[f"{key}_epoch{i}"] = epoch
            y_epochs[f"{key}_epoch{i}"] = y_dict[key]

    return x_epochs, y_epochs


def kfold_split(x_epochs, y_epochs, n_splits=5, shuffle=True, random_state=None):
    """
    Perform k-fold cross-validation on a dataset of EEG epochs.
    Parameters
    ----------
    x_epochs : dict
        A dictionary of EEG epochs, where each key is an epoch ID and each value is a numpy array of shape (n_channels, n_samples).
    y_epochs : dict
        A dictionary of target labels for each epoch, where each key is an epoch ID and each value is an integer target label.
    n_splits : int, optional
        The number of folds to create in the cross-validation. Default is 5.
    shuffle : bool, optional
        Whether to shuffle the data before creating folds. Default is True.
    random_state : int or RandomState, optional
        If an integer, `random_state` is the seed used by the random number generator. If a RandomState instance, `random_state` is the random number generator. If None, the random number generator is the RandomState instance used by `np.random`. Default is None.
    Returns
    -------
    train_epochs : list of dicts
        A list of training epochs for each fold, where each dictionary is of the same format as `x_epochs`.
    test_epochs : list of dicts
        A list of test epochs for each fold, where each dictionary is of the same format as `x_epochs`.
    train_labels : list of dicts
        A list of target labels for the training epochs for each fold, where each dictionary is of the same format as `y_epochs`.
    test_labels : list of dicts
        A list of target labels for the test epochs for each fold, where each dictionary is of the same format as `y_epochs`.
    """

    subject_ids = np.unique(['_'.join(k.split('_')[:-1]) for k in x_epochs.keys()])
    y_subjects = np.array([y_epochs[f'{subject_id}_epoch0'] for subject_id in subject_ids])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    train_epochs = []
    train_labels = []
    test_epochs = []
    test_labels = []
    for train_subjects, test_subjects in skf.split(subject_ids, y_subjects):
        train_subjects_list = subject_ids[train_subjects]
        test_subjects_list = subject_ids[test_subjects]
        train_epochs.append({k: v for k, v in x_epochs.items() if '_'.join(k.split('_')[:-1]) in train_subjects_list})
        train_labels.append({k: v for k, v in y_epochs.items() if '_'.join(k.split('_')[:-1]) in train_subjects_list})
        test_epochs.append({k: v for k, v in x_epochs.items() if '_'.join(k.split('_')[:-1]) in test_subjects_list})
        test_labels.append({k: v for k, v in y_epochs.items() if '_'.join(k.split('_')[:-1]) in test_subjects_list})

    return train_epochs, test_epochs, train_labels, test_labels

