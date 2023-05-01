import numpy as np

from utils.data import extract_eeg_data, extract_psd_data, multi_to_binary_classification, split_dataset, dict_to_arr, epoch_data_and_labels
from utils.labels import get_stai_labels, get_pss_labels
from utils.valid_recs import get_valid_recs
import utils.variables as v




def load_data(data_type, label_type, epoched = False, binary = True):

    # Loads valid recording into valid_recs
    valid_recs = get_valid_recs(data_type=data_type, output_type = 'np')
    #print(f'Valid recs: \n {valid_recs}')

    # Loads EEG data into x_dict_
    x_dict_ = extract_eeg_data(valid_recs, data_type=data_type, output_type='np')
    print(f"\nLength of data: {len(x_dict_)}")

    # Loads correct labels into y_dict_
    if label_type == 'stai':
        y_dict_ = get_stai_labels(valid_recs) 
    elif label_type == 'pss':
        y_dict_ = get_pss_labels(valid_recs)
    else:
        print('No such label type in data set')

    print(f"\nLength of data after removing invalid labels: {len(x_dict_)}")
    print(f"Length of labels after removing invalid labels: {len(y_dict_)}")   

    # Default: Changes the data into binary classes
    if binary:
        x_dict, y_dict = multi_to_binary_classification(x_dict_, y_dict_)
        print(f"\nLength of data after removing mildly stressed subjects: {len(x_dict_)}")
        print(f"Length of labels after removing  mildly stressed subjects: {len(y_dict_)}")
    else:
        # Or keeps the three classes. Must be specified by "binary" parameter
        x_dict = x_dict_.copy()
        y_dict = y_dict_.copy()

    # Splits dataset into train, test, validation
    train_data_dict, test_data_dict, val_data_dict, train_labels_dict, test_labels_dict, val_labels_dict = split_dataset(x_dict, y_dict, kfold = False)
    print(f"\nLength of train data set: {len(train_data_dict)}")
    print(f"Length of validation data set: {len(val_data_dict)}")
    print(f"Length of test data set: {len(test_data_dict)}")

    train_data = dict_to_arr(train_data_dict, data_type)
    test_data = dict_to_arr(test_data_dict, data_type)
    val_data = dict_to_arr(val_data_dict, data_type)
    
    train_labels = np.reshape(np.array(list(train_labels_dict.values())), (len(train_data),1))
    test_labels = np.reshape(np.array(list(test_labels_dict.values())), (len(test_data),1))
    val_labels = np.reshape(np.array(list(val_labels_dict.values())), (len(val_data),1))

    if epoched:
        if data_type == 'new_ica':
            train_data, train_labels = epoch_data_and_labels(train_data, train_labels, sfreq=v.NEW_SFREQ)
            test_data, test_labels = epoch_data_and_labels(test_data, test_labels, sfreq=v.NEW_SFREQ)
            val_data, val_labels = epoch_data_and_labels(val_data, val_labels, sfreq=v.NEW_SFREQ)
        else:
            train_data, train_labels = epoch_data_and_labels(train_data, train_labels, sfreq=v.SFREQ)
            test_data, test_labels = epoch_data_and_labels(test_data, test_labels, sfreq=v.SFREQ)
            val_data, val_labels = epoch_data_and_labels(val_data, val_labels, sfreq=v.SFREQ)        

    print(f"\nShape of train data set: {train_data.shape}")
    print(f"Shape of train labels set: {train_labels.shape}")

    print(f"Shape of validation data set: {val_data.shape}")
    print(f"Shape of validation labels set: {val_labels.shape}")

    print(f"Shape of test data set: {test_data.shape}")
    print(f"Shape of test labels set: {test_labels.shape}")

    return train_data, test_data, val_data, train_labels, test_labels, val_labels


def load_kfold_data(data_type, label_type, epoched = False, binary = True):
     # Loads valid recording into valid_recs
    valid_recs = get_valid_recs(data_type=data_type, output_type = 'np')
    #print(f'Valid recs: \n {valid_recs}')

    # Loads EEG data into x_dict_
    x_dict_ = extract_eeg_data(valid_recs, data_type=data_type, output_type='np')
    print(f"\nLength of data: {len(x_dict_)}")

    # Loads correct labels into y_dict_
    if label_type == 'stai':
        y_dict_ = get_stai_labels(valid_recs) 
    elif label_type == 'pss':
        y_dict_ = get_pss_labels(valid_recs)
    else:
        print('No such label type in data set')

    print(f"\nLength of data after removing invalid labels: {len(x_dict_)}")
    print(f"Length of labels after removing invalid labels: {len(y_dict_)}")   

    # Default: Changes the data into binary classes
    if binary:
        x_dict, y_dict = multi_to_binary_classification(x_dict_, y_dict_)
        print(f"\nLength of data after removing mildly stressed subjects: {len(x_dict_)}")
        print(f"Length of labels after removing  mildly stressed subjects: {len(y_dict_)}")
    else:
        # Or keeps the three classes. Must be specified by "binary" parameter
        x_dict = x_dict_.copy()
        y_dict = y_dict_.copy()

    train_data_dict, test_data_dict, train_labels_dict, test_labels_dict = split_dataset(x_dict, y_dict, kfold = True)
    train_data_arr = dict_to_arr(train_data_dict, data_type)
    test_data_arr = dict_to_arr(test_data_dict, data_type)

    train_labels_arr = np.reshape(np.array(list(train_labels_dict.values())), (len(train_data_arr),1))
    test_labels_arr = np.reshape(np.array(list(test_labels_dict.values())), (len(test_data_arr),1))
    
    if epoched:
        if data_type == 'new_ica':
            train_data, train_labels = epoch_data_and_labels(train_data_arr, train_labels_arr, sfreq=v.NEW_SFREQ)
            test_data, test_labels = epoch_data_and_labels(test_data_arr, test_labels_arr, sfreq=v.NEW_SFREQ)
        else:
            train_data, train_labels = epoch_data_and_labels(train_data_arr, train_labels_arr, sfreq=v.SFREQ)  
            test_data, test_labels = epoch_data_and_labels(test_data_arr, test_labels_arr, sfreq=v.SFREQ)
    else:
        train_data, train_labels = train_data_arr.copy(), train_labels_arr.copy()
        test_data, test_labels = test_data_arr.copy(), test_labels_arr.copy()  

    print(f"\nShape of train data set: {train_data.shape}")
    print(f"Shape of train labels set: {train_labels.shape}")
    print(f"Shape of test data set: {test_data.shape}")
    print(f"Shape of test labels set: {test_labels.shape}")

    return train_data, test_data, train_labels, test_labels



def load_psd_data(label_type, binary = True):

    data_type = 'psd'
    # Loads valid recording into valid_recs
    valid_recs = get_valid_recs(data_type=data_type, output_type = 'np')
    #print(f'Valid recs: \n {valid_recs}')

    # Loads EEG data into x_dict_
    x_dict_ = extract_psd_data(valid_recs)
    print(f"\nLength of data: {len(x_dict_)}")

    # Loads correct labels into y_dict_
    if label_type == 'stai':
        y_dict_ = get_stai_labels(valid_recs) 
    elif label_type == 'pss':
        y_dict_ = get_pss_labels(valid_recs)
    else:
        print('No such label type in data set')

    print(f"\nLength of data after removing invalid labels: {len(x_dict_)}")
    print(f"Length of labels after removing invalid labels: {len(y_dict_)}")   

    # Default: Changes the data into binary classes
    if binary:
        x_dict, y_dict = multi_to_binary_classification(x_dict_, y_dict_)
        print(f"\nLength of data after removing mildly stressed subjects: {len(x_dict_)}")
        print(f"Length of labels after removing  mildly stressed subjects: {len(y_dict_)}")
    else:
        # Or keeps the three classes. Must be specified by "binary" parameter
        x_dict = x_dict_.copy()
        y_dict = y_dict_.copy()

    train_data_dict, test_data_dict, train_labels_dict, test_labels_dict = split_dataset(x_dict, y_dict, kfold = True)
    train_data_arr = dict_to_arr(train_data_dict, data_type)
    test_data_arr = dict_to_arr(test_data_dict, data_type)

    train_labels_arr = np.reshape(np.array(list(train_labels_dict.values())), (len(train_data_arr),1))
    test_labels_arr = np.reshape(np.array(list(test_labels_dict.values())), (len(test_data_arr),1))
    
    train_data, train_labels = train_data_arr.copy(), train_labels_arr.copy()
    test_data, test_labels = test_data_arr.copy(), test_labels_arr.copy()  

    print(f"\nShape of train data set: {train_data.shape}")
    print(f"Shape of train labels set: {train_labels.shape}")
    print(f"Shape of test data set: {test_data.shape}")
    print(f"Shape of test labels set: {test_labels.shape}")

    return train_data, test_data, train_labels, test_labels