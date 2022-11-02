import scipy
import os
import numpy as np
import pandas as pd

DIR_RAW = 'Data/raw_data'
DIR_FILTERED = 'Data/filtered_data'
DIR_ICA_FILTERED = 'Data/ica_filtered_data'

LABELS_PATH = 'Data/scales.xls'

COLUMNS_TO_RENAME = {
    'Subject No.': 'subject_no',
    'Trial_1': 't1_math',
    'Unnamed: 2': 't1_symmetry',
    'Unnamed: 3': 't1_stroop',
    'Trial_2': 't2_math',
    'Unnamed: 5': 't2_symmetry',
    'Unnamed: 6': 't2_stroop',
    'Trial_3': 't3_math',
    'Unnamed: 8': 't3_symmetry',
    'Unnamed: 9': 't3_stroop'
}


def load_dataset(raw=True, test_type="Arithmetic"):
    '''
    Loads data from the SAM 40 Dataset with the test specified by test_type.
    The raw flag specifies whether to use the raw data or the filtered data.
    Returns a Numpy Array with shape (120, 32, 3200).
    '''
    if raw:
        dir = DIR_RAW
        data_key = 'Data'
    else:
        dir = DIR_FILTERED
        data_key = 'Clean_data'

    dataset = np.empty((120, 32, 3200))

    counter = 0
    for filename in os.listdir(dir):
        if test_type not in filename:
            continue

        f = os.path.join(dir, filename)
        data = scipy.io.loadmat(f)[data_key]
        dataset[counter] = data
        counter += 1
    return dataset


def load_ica_dataset(test_type="Arithmetic"):
    dir = DIR_ICA_FILTERED
    data_key = 'Clean_data'

    dataset = np.empty((120, 32, 3200))

    counter = 0
    for filename in os.listdir(dir):
        if test_type not in filename:
            continue

        f = os.path.join(dir, filename)
        data = scipy.io.loadmat(f)[data_key]
        data = np.reshape(data,(32,3200))
        dataset[counter] = data
        counter += 1
    return dataset

def load_labels():
    labels = pd.read_excel(LABELS_PATH)
    labels = labels.rename(columns=COLUMNS_TO_RENAME)
    labels = labels[1:]
    labels = labels.astype("int")
    labels = labels > 5
    return labels


def convert_to_epochs(dataset, channels, sfreq):
    '''
    Splits EEG data into epochs with length specified by epoch_length
    '''
    epoched_dataset = np.empty(
        (dataset.shape[0], dataset.shape[2]//sfreq, channels, sfreq))
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[2]//sfreq):
            epoched_dataset[i, j] = dataset[i, :, j*sfreq:(j+1)*sfreq]
    return epoched_dataset
