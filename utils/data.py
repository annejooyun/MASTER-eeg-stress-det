import mne
import os
import numpy as np
import scipy
from sklearn.model_selection import StratifiedKFold, train_test_split
import logging
import utils.variables as v


def read_eeg_data(data_type, filename, output_type):

    if data_type == 'raw':
        data_key = 'raw_eeg_data'
    elif data_type == 'ica':
        data_key = 'Clean_data'
    else:
        print(f'No data with data_type = {data_type} found')
        return 0
    
    data = scipy.io.loadmat(filename)[data_key]
    data = data[:,:75000] # 5 MIN * 60 SEK/MIN * 250 SAMPLES/SEK = 75 000 SAMPLES

    if output_type == 'np':
        return data
    
    elif output_type == 'mne':
        info = mne.create_info(8, sfreq=v.SFREQ, ch_types= 'eeg', verbose=None)
        raw_arr = mne.io.RawArray(data, info) 
        mne.rename_channels(raw_arr.info, v.MAPPING)
        return raw_arr
    else:
        print(f'No data with output_type = {output_type} found')
        return 0


def extract_eeg_data(valid_recs, data_type, output_type):
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
        f_name = os.path.join(dir, f'sub-{subject}_ses-{session}_run-{run}.mat')
        try:
            data = read_eeg_data(data_type, f_name, output_type)
        except:
            logging.error(f"2) Failed to read data for recording {rec}")
            data = None

        key = f"{subject}_{session}_{run}"
        eeg_data[key] = data
    return eeg_data


def multi_to_binary_classification(x_dict, y_dict):
    targ_val = 1
    rem=[]
    for i in y_dict.keys():
        if y_dict[i] is targ_val:
            rem.append(i)
    # printing result
    print("\nThe extracted keys : \n" + str(rem))

    [y_dict.pop(key) for key in rem]
    [x_dict.pop(key) for key in rem]
    print(f"\nDictionary after removal of keys from y_dict: \n {y_dict.keys()}")
    print(f"\nDictionary after removal of keys from x_dict: \n {x_dict.keys()}")
    return x_dict, y_dict

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


def split_dataset(x_dict, y_dict):
    """
    We split the dataset into training-, validation- and test-sets.

    - Training consists of 60% of the participants
    - Validation consists of 20% of the participants
    - Test consists of 20% of the participants
    """

    keys_list = list(x_dict.keys())

    subject_list = []
    for i in range(v.NUM_SUBJECTS+1):
        subject = f'P{str(i).zfill(3)}'
        for key in keys_list:
            if subject in key and subject not in subject_list:
                subject_list.append(subject)

    mean_labels_list = []
    for i in range(v.NUM_SUBJECTS+1):
        sum_label = 0
        num_recordings = 0
        subject = f'P{str(i).zfill(3)}'
        for key, value in y_dict.items():
            if subject in key:
                sum_label += value
                num_recordings += 1
        if num_recordings == 0:
            continue
        else:
            mean_label = sum_label/num_recordings
            mean_labels_list.append(round(mean_label,0))

    subjects, subjects_test, mean_labels, mean_labels_test = train_test_split(subject_list, mean_labels_list, test_size= 0.2, random_state=42, stratify = mean_labels_list)
    subjects_train, subjects_val, mean_labels_train, mean_labels_val = train_test_split(subjects, mean_labels, test_size=0.25, random_state=42, stratify = mean_labels)
        
    train_data_dict, train_labels_dict = reconstruct_dicts(subjects_train, x_dict, y_dict)
    test_data_dict, test_labels_dict = reconstruct_dicts(subjects_test, x_dict, y_dict)
    val_data_dict, val_labels_dict = reconstruct_dicts(subjects_val, x_dict, y_dict)

    return train_data_dict, test_data_dict, val_data_dict, train_labels_dict, test_labels_dict, val_labels_dict


def reconstruct_dicts(subjects_list, x_dict, y_dict):
    data_dict = {}
    labels_dict = {}

    for subject in subjects_list:
        # Reconstructing data dict
        for key, val in x_dict.items():
            if subject in key:
                data_dict[key] = val

        #Reconstructing labels dict
        for key, val in y_dict.items():
            if subject in key:
                labels_dict[key] = val

    return(data_dict, labels_dict)
