import os


DIR_RAW = 'Data/Raw_eeg'
DIR_128HZ_RAW = 'Data/128Hz_raw_eeg'
DIR_ICA_FILTERED = 'Data/ICA_data'
DIR_INIT_FILTERED = 'Data/Init_filter_data'
DIR_NEW_ICA = 'Data/New_ICA_data'
DIR_PSD = 'Data/PSD_data'

LABELS_PATH = 'utils/STAI_grading.xlsx'

NUM_SUBJECTS = 28
NUM_SESSIONS = 2
NUM_RUNS = 2
NUM_CHANNELS = 8
NUM_SAMPLES = 75000
SFREQ = 250

# For PSD_data
NUM_PSD_FREQS = 129

# For New_ICA_data
NEW_SFREQ = 128
NEW_NUM_SAMPLES = 38400

#For downsampled raw
DOWNSAMPLED_SFREQ = 128

EPOCH_LENGTH = 0.1


DATA_TYPES = ['raw','128Hz_raw', 'ica', 'init', 'new_ica', 'psd']
MAPPING = {'0':'F4','1':'Fp2','2':'C3','3':'FC6','4':'O1','5':'Oz','6':'FT9','7':'T8'}