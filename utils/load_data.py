import numpy as np

from utils.data import extract_eeg_data, extract_psd_data, multi_to_binary_classification, split_dataset, dict_to_arr, epoch_data_and_labels
from utils.labels import get_stai_labels, get_ss_labels
from utils.valid_recs import get_valid_recs
import utils.variables as v
import utils.features as f
import utils.load_SAM40_data as ld_SAM40



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
        y_dict_ = get_ss_labels(valid_recs)
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
        y_dict_ = get_ss_labels(valid_recs)
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
        y_dict_ = get_ss_labels(valid_recs)
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




def load_and_shape_data(data_type, label_type, feature, kfold, new_ica = False):
    #Load data
    if kfold:
        train_data, test_data, train_labels, test_labels = load_kfold_data(data_type, label_type, epoched = False, binary = True)
    else:
        train_data, test_data, val_data, train_labels, test_labels, val_labels = load_data(data_type, label_type, epoched = True, binary = True)
        return train_data, test_data, val_data, train_labels, test_labels, val_labels
    
    print('\n---- Balanced dataset? ----')
    print(f'Section of non-stressed in train set: {np.sum(train_labels == 0)/len(train_labels)}')
    print(f'Section of non-stressed in test set: {np.sum(test_labels == 0)/len(test_labels)}')


    if feature:
        #Reshape labels to fit (n_recordings*n_channels, 1)
        train_labels = np.repeat(train_labels, repeats = v.NUM_CHANNELS, axis = 0).reshape((train_data.shape[0]*v.NUM_CHANNELS,1))
        train_labels = train_labels.ravel()

        test_labels = np.repeat(test_labels,repeats = v.NUM_CHANNELS, axis = 0).reshape((test_data.shape[0]*v.NUM_CHANNELS,1))
        test_labels = test_labels.ravel()
        
        #Extract features
        #time_series_features, fractal_features, entropy_features, hjorth_features, freq_band_features, kymatio_wave_scattering
        train_data = f.time_series_features(train_data, new_ica)
        test_data = f.time_series_features(test_data, new_ica)

        return train_data, test_data, train_labels, test_labels
    else:
        #Reshape data
        train_data = np.reshape(train_data, (train_data.shape[0]*train_data.shape[1], train_data.shape[2]))
        train_labels = np.repeat(train_labels, repeats = 8, axis = 1).reshape(-1,1)
        train_labels = train_labels.ravel()

        test_data = np.reshape(test_data, (test_data.shape[0]*test_data.shape[1],test_data.shape[2]))
        test_labels = np.repeat(test_labels, repeats = 8, axis = 1).reshape(-1,1)
        test_labels = test_labels.ravel()
        return train_data, test_data, train_labels, test_labels

def load_and_shape_psd_data(label_type):
    train_data, test_data, train_labels, test_labels = ld.load_psd_data(label_type, binary = True)
    train_data = np.reshape(train_data, (train_data.shape[0]*train_data.shape[1], train_data.shape[2]))
    train_labels = np.repeat(train_labels, repeats = 8, axis = 1).reshape(-1,1)
    train_labels = train_labels.ravel()

    test_data = np.reshape(test_data, (test_data.shape[0]*test_data.shape[1],test_data.shape[2]))
    test_labels = np.repeat(test_labels, repeats = 8, axis = 1).reshape(-1,1)
    test_labels = test_labels.ravel()
    
    return train_data, test_data, train_labels, test_labels

def load_and_shape_SAM40_data():
    #Load the SAM40 dataset to be used as test data/label
    selected_channels_names = ['Fp2', 'F4', 'FC6', 'T8', 'Oz', 'O1', 'C3', 'FT9']
    dataset_SAM40_ = ld_SAM40.load_dataset('raw', 'Arithmetic')
    channels = ld_SAM40.load_channels()

    #Extract only the same channels
    selected_chan_index = [channels.index(elem) for elem in selected_channels_names]
    selected_channels_dataset = np.array([dataset_SAM40_[:,i,:] for i in selected_chan_index])
    selected_channels_dataset = np.reshape(selected_channels_dataset, (120,8,3200))

    #Compute time_series_features for SAM40
    test_data_SAM40 = f.time_series_features(selected_channels_dataset, False, SAM40=True)

    #Load SAM40 labels 
    labels_ = ld_SAM40.load_labels()
    labels = pd.concat([labels_['t1_math'], labels_['t2_math'],
                    labels_['t3_math']]).to_numpy()
    
    #Change labels from T/F to 1/0
    for i in range(len(labels)):
        if labels[i]:
            labels[i] = 1
        else:
            labels[i] = 0

    test_labels_SAM40 = np.repeat(labels, test_data_SAM40.shape[0]//labels.shape[0])

    print(f'SAM40 test data shape: {test_data_SAM40.shape}')
    print(f'SAM40 test labels shape: {test_labels_SAM40.shape}')
    return test_data_SAM40, test_labels_SAM40