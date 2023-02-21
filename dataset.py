import scipy
import os
import numpy as np
import pandas as pd
import variables as v


def load_dataset(data_type="ica"):
    '''
    Loads data from the dataset.
    The data_type parameter specifies which of the datasets to load. Possible values
    are raw, filtered, ica_filtered.
    Returns a Numpy Array with shape (120, 32, 3200).
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

    dataset = np.empty((v.NUM_SUBJECTS, v.NUM_CHANNELS, v.SFREQ*5*60)) #5 minute recorings

    counter = 0
    for filename in os.listdir(dir):
        if test_type not in filename:
            continue

        f = os.path.join(dir, filename)
        data = scipy.io.loadmat(f)[data_key]
        dataset[counter] = data
        counter += 1
    return dataset


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
        for j in range(v.N_SESSIONS*v.N_RUNS):

            label = int(scores.iloc[i, j+8]
                        in range(low_cutoff, high_cutoff+1))
            if scores.iloc[i, j+8] > high_cutoff:
                label = 2
            subject = i + 1
            session = j//v.N_RUNS + 1
            run = j % v.N_RUNS + 1
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
        coordinates_file = os.path.join(root,"Coordinates.locs") 

        channel_names = []

        with open(coordinates_file, "r") as file:
            for line in file:
                elements = line.split()
                channel = elements[-1]
                channel_names.append(channel)
                
        return channel_names