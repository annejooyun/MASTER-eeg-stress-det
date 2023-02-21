import mne
import os
import numpy as np
import pandas as pd
import variables as v
from sklearn.model_selection import StratifiedKFold
import logging



def extract_eeg_data(valid_recs, data_type="raw"):
    """
    Extract EEG data from a list of valid recordings.
    Parameters
    ----------
    valid_recs : list
        List of valid recording names.
    data_type : str, optional
        Data type to be extracted, either "raw" or "filtered", by default "filtered"
    Returns
    -------
    dict
        Dictionary containing EEG data with keys as recording names.
    """

    if data_type == "raw":
        dir = v.DIR_RAW
    elif data_type == "filtered":
        dir = v.DIR_FILTERED
    else:
        return
    eeg_data = {}
    for rec in valid_recs:
        subject, session, run = rec.split('_')
        f_name = os.path.join(
            dir, f'sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-Default_run-{run}_eeg.fif')
        try:
            data = read_eeg_data(f_name)
        except:
            logging.error(f"Failed to read data for recording {rec}")
            data = None
        key = f"{subject}_{session}_{run}"
        eeg_data[key] = data
    return eeg_data


def read_eeg_data(filename):
    """
    Read EEG data from a file.
    Parameters
    ----------
    filename : str
        Path to the file to be read.
    Returns
    -------
    data : instance of Raw object
        The EEG data contained in the file.
    """
    return mne.io.read_raw_fif(filename)


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


def filter_score_labels(valid_recs, labels):
    """
    Filter the labels dictionary to only contain labels for valid recordings.
    Parameters
    ----------
    valid_recs : list
        List of valid recordings.
    labels : dict
        Dictionary containing labels for all recordings.
    Returns
    -------
    dict
        Dictionary containing labels for valid recordings.
    """
    filtered_labels = {}
    for rec in valid_recs:
        filtered_labels[rec] = labels[rec]
    return filtered_labels


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
    return filter_score_labels(valid_recs, labels)


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

valid_recs = get_valid_recs()
x_dict = extract_eeg_data(valid_recs)
scores = compute_average_scores()
y_dict = get_labels(valid_recs)

x_epochs, y_epochs = extract_epochs(x_dict, y_dict, epoch_duration=3)
splits = kfold_split(x_epochs, y_epochs, n_splits=5, shuffle=True, random_state=42)
train_epochs, test_epochs, train_labels, test_labels = splits

print(len(train_labels[0]))