import numpy as np
import pandas as pd
import logging
import math

import utils.variables as v

def load_pss_scores(filename):
    """
    Load and binarize PSS (Perceived Stress Scale) scores from an Excel file.
    Parameters
    ----------
    filename : str
        The path to the Excel file containing the PSS scores.

    Returns
    -------
    scores : pandas.DataFrame
        A DataFrame containing the PSS scores, where each row represents a subject and each
        column represents a session. 
        Scores are represented as:
         - 0/non-stressed for values between 0-3, 
         - 1/mildly-stressed for values betweem 4-6 and
         - 2/stressed for values between 7-10
    """
    scores = pd.read_excel(filename, sheet_name='Rating 1-10')
    return scores


def filter_pss_labels(scores, valid_recs, low_cutoff, high_cutoff):
    """
    Filter PSS (Perceived Stress Scale) scores by record ID.
    Parameters
    ----------
    scores : pandas.DataFrame
        A DataFrame containing PSS scores for each subject and session.
    valid_recs : list of str
        A list of valid record IDs to keep in the filtered scores.
    Returns
    -------
    filtered_labels : dict
        A dictionary containing the filtered PSS scores, where the keys are record IDs and the values are the corresponding scores.
    """
    #Turning scores lower than low_cutoff to label = 0
    #               higher than high_cutoff to label = 2
    #               the rest to label = 1
    print(scores.iloc[:, 1:])
    scores.iloc[:, 1:] = scores.iloc[:, 1:].applymap(
        lambda x: 0 if pd.isna(x) else (0 if x <= low_cutoff else (2 if x >= high_cutoff else 1)))
    
    labels = {}
    for i in range(v.NUM_SUBJECTS):
        for j in range(v.NUM_SESSIONS*v.NUM_RUNS):
            subject = i + 1
            session = math.ceil((j+1)/v.NUM_SESSIONS)
            run = j%v.NUM_RUNS + 1

            key = f'P{str(subject).zfill(3)}_S{str(session).zfill(3)}_{str(run).zfill(3)}'
            if key in valid_recs:
                row = scores.loc[i]
                labels[key] = row[f'S{str(session).zfill(3)}_{str(run).zfill(3)}']
            else:
                print(f"{key} has invalid record length")

    print("\n---- Labels ----")
    print(labels)
    return labels


def get_pss_labels(valid_recs, filename=v.LABELS_PATH, low_cutoff = 3, high_cutoff = 7):
    """
    Get filtered and binarized PSS (Perceived Stress Scale) scores for a list of valid record IDs.
    Parameters
    ----------
    valid_recs : list of str
        A list of valid record IDs to keep in the filtered scores.
    filename : str, optional
        The path to the Excel file containing the PSS scores. Default is v.LABELS_PATH.
    threshold : int or float, optional
        The threshold value to use for binarizing the PSS scores. Scores greater than `threshold` will be
        binarized to 1, and scores less than or equal to `threshold` will be binarized to 0. Default is 4.
    Returns
    -------
    filtered_scores : dict
        A dictionary containing the filtered and binarized PSS scores, where the keys are record IDs and the
        values are the corresponding scores.
    """
    scores = load_pss_scores(filename)
    filtered_labels = filter_pss_labels(scores, valid_recs, low_cutoff, high_cutoff)
    return filtered_labels





#-----------------------------------------------------------------------------------------------------


def compute_stai_y1_scores(path=v.LABELS_PATH):
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
    columns = ['SubjectNo', 'D1Y1', 'D2Y1', 'J1Y1', 'J2Y1']
    
    # Open file
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

        y1_indices = sheet[sheet.columns[0]] == 'Total score Y1'
        y1_scores = sheet.loc[y1_indices].values[0][1::2]

        scores.append(np.concatenate([y1_scores]))
    scores_df = pd.DataFrame(scores, columns=columns[1:])


    scores_df.insert(0, columns[0], range(1, n_subjects+1))
    return scores_df


def compute_stai_labels(scores, valid_recs, low_cutoff, high_cutoff):
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
    pd.DataFrame
        Dataframe containing the labels (0, 1 or 2) for all subjects.
    """

    #Turning scores lower than low_cutoff to label = 0
    #               higher than high_cutoff to label = 2
    #               the rest to label = 1
    scores.iloc[:, 1:] = scores.iloc[:, 1:].applymap(
        lambda x: 0 if x < low_cutoff else (2 if x > high_cutoff else 1))

    #Only keeping valid recordings
    labels = {}
    for i in range(v.NUM_SUBJECTS):
        for j in range(v.NUM_SESSIONS*v.NUM_RUNS):
            subject = i + 1
            session = math.ceil((j+1)/v.NUM_SESSIONS)
            run = j%v.NUM_RUNS + 1

            key = f'P{str(subject).zfill(3)}_S{str(session).zfill(3)}_{str(run).zfill(3)}'
            if key in valid_recs:
                labels[key] = scores.iloc[i,j+1]
            else:
                print(f"{key} has invalid record length")

    print("\n---- Labels ----")
    print(labels)
    return labels


def get_stai_labels(valid_recs, path=v.LABELS_PATH, low_cutoff=37, high_cutoff=45):
    """
    Get labels for valid recordings.
    Parameters
    ----------
    valid_recs : list
        List of valid recordings.
    path : str, optional
        Path to the raw scores excel file. Default is v.LABELS_PATH.
    low_cutoff : int, optional
        The lower cutoff value. Default is 37.
    high_cutoff : int, optional
        The upper cutoff value. Default is 45.
    Returns
    -------
    dict
        Dictionary containing labels for valid recordings.
    """
    scores = compute_stai_y1_scores(path)
    labels = compute_stai_labels(scores, valid_recs, low_cutoff, high_cutoff)
    return labels
