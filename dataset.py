import scipy
import os
import numpy as np
import pandas as pd
import variables as v

def read_eeg_data(dir, filename):
    data_key = 'raw_eeg_data'
    f = dir + filename
    return scipy.io.loadmat(f)[data_key]

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


def filter_valid_recs(recs):
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

    dir = v.DIR_RAW
    valid_recs = []
    for rec in recs:
        subject, session, run = rec.split('_')
        f_name = os.path.join(
            dir, f'sub-{subject}_ses-{session}_run-{run}.mat')
        try:
            data = read_eeg_data(f_name)
            if data.n_times / data.info['sfreq'] >= 4.30 * 60:
                valid_recs.append(rec)
        except:
            logging.error(f"Failed to read data for recording {rec}")
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

    #dataset = np.empty((v.NUM_SUBJECTS, v.NUM_CHANNELS, v.SFREQ*5*60)) #5 minute recorings
    dataset = {}

    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(f)
        key = 
        dataset


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