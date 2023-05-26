import mne
import os
import numpy as np
import scipy
from sklearn.model_selection import StratifiedKFold, train_test_split

import utils.variables as v

def read_eeg_data(data_type, filename, output_type):
    '''
    Loads eeg data and returns the appropriate output dependant on output_type
    '''
    #Assoociating correct data_key to the inputted data_type
    if data_type == 'raw':
        data_key = 'raw_eeg_data'
    elif data_type == '128Hz_raw':
        data_key = 'downsampled_raw'
    elif data_type== 'ica' or data_type == 'init' or data_type == 'new_init' or data_type == 'new_ica':
        data_key = 'Clean_data'
    else:
        print(f'No data with data_type = {data_type} found')
        return 0
    
    #Loads data
    data = scipy.io.loadmat(filename)[data_key]

    #Removes data recorded after 5 minutes
    data = data[:,:75000] # 5 MIN * 60 SEK/MIN * 250 SAMPLES/SEK = 75 000 SAMPLES
    #data = data[:,:3200] # 25 SEK/MIN * 128 SAMPLES/SEK = 3200 SAMPLES

    #Checks output type and returns correct data
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
    are raw, ica filtered and initially filtered.
    Returns
    '''
    assert (data_type in v.DATA_TYPES)

    if data_type == 'raw':
        dir = v.DIR_RAW
    elif data_type == '128Hz_raw':
        dir = v.DIR_128HZ_RAW
    elif data_type == 'ica':
        dir = v.DIR_ICA_FILTERED
    elif data_type == 'init':
        dir = v.DIR_INIT_FILTERED
    elif data_type == 'new_init':
        dir = v.DIR_NEW_INIT_FILTERED
    elif data_type == 'new_ica':
        dir = v.DIR_NEW_ICA
    else:
        print("No files matching data type found")
        return 0

    eeg_data = {}
    for rec in valid_recs:
        subject, session, run = rec.split('_')
        f_name = os.path.join(dir, f'sub-{subject}_ses-{session}_run-{run}.mat')
        try:
            data = read_eeg_data(data_type, f_name, output_type)
            key = f"{subject}_{session}_{run}"
            eeg_data[key] = data
        except:
            print(f"ERROR 2) Failed to read data for recording {rec}")
            data = None
    return eeg_data

def read_psd_data(filename):
    return scipy.io.loadmat(filename)['psd_data']
    
def extract_psd_data(valid_recs):
    '''
    Loads psd data from the dataset.
    '''
    dir = v.DIR_PSD

    psd_data = {}
    for rec in valid_recs:
        subject, session, run = rec.split('_')
        f_name = os.path.join(dir, f'sub-{subject}_ses-{session}_run-{run}.mat')
        try:
            data = read_psd_data(f_name)
            key = f"{subject}_{session}_{run}"
            psd_data[key] = data
        except:
            print(f"ERROR 2) Failed to read data for recording {rec}")
            data = None
    return psd_data

def multi_to_binary_classification(x_dict, y_dict):
    '''
    Removes mildly stressed recordings, thus changing the target values from multi to binary.
    '''
    targ_val = 1
    remove = []
    for key in y_dict.keys():
        if y_dict[key] == targ_val:
            remove.append(key)

    #Printing result
    print("\nThe extracted keys : \n" + str(remove))

    #Removes the keys in yhe list remove 
    [y_dict.pop(key) for key in remove]
    [x_dict.pop(key) for key in remove]

    #Changing labels from 2 to 1
    for key in y_dict.keys():
        if y_dict[key] == 2:
            y_dict[key] = 1

    return x_dict, y_dict

# Christians
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

#Christians
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


def split_dataset(x_dict, y_dict, kfold = True):
    """
    Splits the dataset into training-, validation- (not for kfold) and test-sets.
    The dataset is split along subjects, not recordings.
    This way it is ensured that one subjects recordings are uniquely in one of the data sets.

    In order to ensure balanced splits, a new "mean label" is calculated for each subject.
    The mean label list is later used as the "stratify" parameter when splitting the dataset.
    
    For kfold:
    - Training consists of 80% of the participants
    - Test consists of 20% of the participants

    Else:
    - Training consists of 60% of the participants
    - Validation consists of 20% of the participants
    - Test consists of 20% of the participants
    """
    print('\n---- Splitting dataset along subjects ----')
    keys_list = list(x_dict.keys())
    
    #Creates a list of all subjects
    subject_list = []
    for i in range(v.NUM_SUBJECTS+1):
        subject = f'P{str(i).zfill(3)}'
        for key in keys_list:
            if subject in key and subject not in subject_list:
                subject_list.append(subject)

    #Calculates a new "mean label" for each participant.
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

    
    if kfold:
        #Dividing subjects between two datasets, as the training set will be used in kfold validation
        print(f'\nSubject list: {subject_list}')
        subjects_train, subjects_test, mean_labels_train, mean_labels_test = train_test_split(subject_list, mean_labels_list, test_size= 0.2, random_state=42, stratify = mean_labels_list)

        #Check no overlapping subjects
        for subject in subjects_train:
            if subject in subjects_test:
                print(f'ERROR: Subject {subject} in both training and test list')    
        
        #Reconstructing train- and test-sets with corresponding data
        train_data_dict, train_labels_dict = reconstruct_dicts(subjects_train, x_dict, y_dict)
        test_data_dict, test_labels_dict = reconstruct_dicts(subjects_test, x_dict, y_dict)

        #Check no overlapping subjects
        for key in train_data_dict.keys():
            if key in test_data_dict.keys():
                print(f'ERROR: Key {key} in both training and test dicts')

        print(f'\nKeys train: {train_data_dict.keys()}')
        print(f'\nKeys test: {test_data_dict.keys()}')

        return train_data_dict, test_data_dict, train_labels_dict, test_labels_dict

    else:
        #Dividing subjects between the three datasets
        print(f'\nSubject list: {subject_list}')
        subjects, subjects_test, mean_labels, mean_labels_test = train_test_split(subject_list, mean_labels_list, test_size= 0.2, random_state=42, stratify = mean_labels_list)
        subjects_train, subjects_val, mean_labels_train, mean_labels_val = train_test_split(subjects, mean_labels, test_size=0.25, random_state=42, stratify = mean_labels)
        
        #Check no overlapping subjects
        for subject in subjects_train:
            if subject in subjects_test:
                print(f'ERROR: Subject {subject} in both training and test list')

        #Reconstructing train-,validation- and test-sets with corresponding data
        train_data_dict, train_labels_dict = reconstruct_dicts(subjects_train, x_dict, y_dict)
        test_data_dict, test_labels_dict = reconstruct_dicts(subjects_test, x_dict, y_dict)
        val_data_dict, val_labels_dict = reconstruct_dicts(subjects_val, x_dict, y_dict)

        #Check no overlapping subjects
        for key in train_data_dict.keys():
            if key in test_data_dict.keys():
                print(f'ERROR: Key {key} in both training and test dicts')

        return train_data_dict, test_data_dict, val_data_dict, train_labels_dict, test_labels_dict, val_labels_dict



def reconstruct_dicts(subjects_list, x_dict, y_dict):
    '''
    Reconstructs the dictionarys after the dataset has been split into train-, validation- and test-sets
    '''
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


def dict_to_arr(data_dict, data_type):
    '''
    Turns dictionary into numpy array
    '''
    keys_list = list(data_dict.keys())
    
    if data_type == 'ica' or data_type == 'raw' or data_type == 'init' or data_type == 'new_init':
        data_arr = np.empty((len(keys_list), v.NUM_CHANNELS, v.NUM_SAMPLES))
    elif data_type == 'new_ica':
        data_arr = np.empty((len(keys_list), v.NUM_CHANNELS, v.NEW_NUM_SAMPLES))
    #elif data_type == '128Hz_raw':
    #    data_arr = np.empty((len(keys_list), v.NUM_CHANNELS, 5*60*128))
    elif data_type == '128Hz_raw':
        data_arr = np.empty((len(keys_list), v.NUM_CHANNELS, 25*128))
    elif data_type == 'psd':
        data_arr = np.empty((len(keys_list), v.NUM_CHANNELS, v.NUM_PSD_FREQS))
    i = 0
    for key in keys_list:
        data = data_dict[key]
        data_arr[i] = data
        i += 1

    return data_arr


def epoch_data_and_labels(data, labels , sfreq = 128):
    
    # Calculate the number of samples per epoch
    samples_per_epoch = int(v.EPOCH_LENGTH * sfreq)

    # Get the shape of the data
    n_recordings, n_channels, n_total_time_steps = data.shape
    print(f'data shape {data.shape}')

    # Calculate the number of epochs
    n_epochs = int(n_total_time_steps // samples_per_epoch)
    print(f'n_epochs {n_epochs}')

    # Create new arrays to hold the epoched data and labels
    data_epoched = np.zeros((int(n_epochs*n_recordings), n_channels, samples_per_epoch))
    labels_epoched = np.zeros((int(n_epochs*n_recordings), 1))

    # Loop over each epoch and extract the corresponding time steps from the data and 
    i = 0
    for rec in range(n_recordings):
        for epoch_idx in range(n_epochs):
            start_idx = epoch_idx * samples_per_epoch
            end_idx = start_idx + samples_per_epoch

            data_slice = data[rec, :, start_idx:end_idx]
            data_epoched[i, :, :] = data_slice

            labels_epoched[i, :] = labels[rec, :]

            i += 1


    return data_epoched, labels_epoched