import scipy
import os
import numpy as np
import pandas as pd
import utils.variables_SAM40 as v_SAM40


def load_dataset(data_type="ica2", test_type="Arithmetic"):
    '''
    Loads data from the SAM 40 Dataset with the test specified by test_type.
    The data_type parameter specifies which of the datasets to load. Possible v_SAM40alues
    are raw, filtered, ica_filtered.
    Returns a Numpy Array with shape (120, 32, 3200).
    '''
    assert (test_type in v_SAM40.TEST_TYPES)

    assert (data_type in v_SAM40.DATA_TYPES)

    dir = v_SAM40.DIR_SAM40
    data_key = 'Data'

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


def load_labels():
    '''
    Loads labels from the dataset and transforms the label v_SAM40alues to binary v_SAM40alues.
    v_SAM40alues larger than 5 are set to 1 and v_SAM40alues lower than or equal to 5 are set to zero.
    '''
    labels = pd.read_excel(v_SAM40.LABELS_PATH)
    labels = labels.rename(columns=v_SAM40.COLUMNS_TO_RENAME)
    labels = labels[1:]
    labels = labels.astype("int")
    labels = labels > 5
    return labels


def convert_to_epochs(dataset, channels, sfreq):
    '''
    Splits EEG data into epochs with length 1 sec
    '''
    epoched_dataset = np.empty((dataset.shape[0], dataset.shape[2]//sfreq, channels, sfreq))
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[2]//sfreq):
            epoched_dataset[i, j] = dataset[i, :, j*sfreq:(j+1)*sfreq]
    return epoched_dataset


def load_channels():
        '''
        Loads the channel names from the file Coordinates.locs
        '''
        root = 'Data'
        coordinates_file = os.path.join(root,"SAM40_Arithmetic/Coordinates.locs") 

        channel_names = []

        with open(coordinates_file, "r") as file:
            for line in file:
                elements = line.split()
                channel = elements[-1]
                channel_names.append(channel)
                
        return channel_names