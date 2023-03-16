import os


DIR_RAW = 'Data/Raw_eeg'
DIR_ICA_FILTERED = 'Data/ICA_data'

LABELS_PATH = 'Data/STAI_grading.xls'

NUM_SUBJECTS = 28
NUM_SESSIONS = 2
NUM_RUNS = 2
NUM_CHANNELS = 8
NUM_SAMPLES = 75000
SFREQ = 250



DATA_TYPES = ["raw", "ica"]
MAPPING = {'0':'F4','1':'Fp2','2':'C3','3':'FC6','4':'O1','5':'Oz','6':'FT9','7':'T8'}