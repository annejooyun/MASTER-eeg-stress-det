import mne
import os
import numpy as np
import scipy
from sklearn.model_selection import StratifiedKFold
import logging
import utils.variables as v


def read_eeg_data(data_type, filename):
    if data_type == 'raw':
        data_key = 'raw_eeg_data'
    elif data_type == 'ica':
        data_key = 'Clean_data'
    else:
        print(f'No data with data_type = {data_type} found')
        return 0
    data = scipy.io.loadmat(filename)[data_key]
    info = mne.create_info(8, sfreq=v.SFREQ, ch_types= 'eeg', verbose=None)
    raw_arr = mne.io.RawArray(data, info)
    mapping = {'0':'F4','1':'Fp2','2':'C3','3':'FC6','4':'O1','5':'Oz','6':'FT9','7':'T8'}
    mne.rename_channels(raw_arr.info, mapping)

    return raw_arr


def extract_eeg_data(valid_recs, data_type):
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

def segment_data(x_dict, y_dict, epoch_duration=3):
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
        events = mne.make_fixed_length_events(raw, stop = 5*60, duration=epoch_duration, overlap=overlap_duration)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=epoch_duration, baseline=None, preload=True)

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


