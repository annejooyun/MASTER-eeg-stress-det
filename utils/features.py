
import mne_features.univariate as mne_f
import mne_features
import numpy as np

import utils.variables as v
import utils.variables_SAM40 as v_SAM40
import numpy as np
import scipy as sp

#Kymatio dependencies
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from kymatio.numpy import Scattering1D, Scattering2D

def kymatio_wave_scattering(data):
    first_key = next(iter(data[0]))
    n_channels, _ = data[0][first_key].shape
    T_ = data[0][first_key].shape[-1]
    #print('T_: ', T_)
    J = 6
    Q = 16
    S = Scattering1D(J,T_,Q)
    x_ = data[0][first_key][0]
    x_ = x_ / np.max(np.abs(x_))
    Sx_ = S(x_)
    #print('Sx_ shape: ', Sx_.shape)

    features_per_channel = Sx_.shape[0]
    #print('n_chan: ', n_channels)
    features = []
    for fold in data:
        n_trials = len(fold)
        #print('ntrials: ',len(fold))
        features_for_fold = np.empty(
            [n_trials, n_channels * features_per_channel])
        for j, key in enumerate(fold):
            trial = fold[key]
            trial = trial / np.max(np.abs(trial))
            #print('trial shape: ', trial.shape)
            T = trial.shape[-1]
            #print('T: ', T)
            J = 6
            Q = 16
            S = Scattering1D(J,T,Q)
            wav_scat = S(trial)
            wav_scat = np.mean(wav_scat, axis=-1)
            wav_scat = np.ndarray.flatten(wav_scat)
            #print('features_for_fold[j] shape: ', features_for_fold[j].shape)
            #print('features_for_fold shape: ', features_for_fold.shape)
            #print('wavscat_t shape: ', np.transpose(wav_scat).shape)
            features_for_fold[j] = np.transpose(wav_scat)
            
        features_for_fold = features_for_fold.reshape(
            [n_trials, n_channels*features_per_channel])
        features.append(features_for_fold)
    return features

def time_series_features(data, new_ica, SAM40 = False):
    '''
    Compute the features peak-to-peak amplitude, variance and rms using the package mne_features.
    The data should be on the form (n_recordings, n_channels, n_samples)
    The output is on the form (n_trials*n_secs, n_channels*3)
    '''
    if SAM40:
        sfreq = v_SAM40.SFREQ
    elif new_ica:
        sfreq = v.NEW_SFREQ
    else:
        sfreq = v.SFREQ
    print('SFREQ: ', sfreq)
    n_recordings = data.shape[0]
    n_samples = data.shape[2]
    n_samples_per_epoch = int(sfreq*v.EPOCH_LENGTH) #n_seconds = 25 250
    print(n_samples_per_epoch)
    n_epochs = int(n_samples/n_samples_per_epoch)
    print(n_epochs)
    
    ptp_amp = np.zeros((n_recordings, v.NUM_CHANNELS, n_epochs))
    variance = np.zeros((n_recordings, v.NUM_CHANNELS, n_epochs))
    rms = np.zeros((n_recordings,v.NUM_CHANNELS,n_epochs))

    for i in range(n_recordings):
        for j in range(v.NUM_CHANNELS):
            for k in range(n_epochs):
                start_indx = k*n_samples_per_epoch
                end_indx = start_indx + n_samples_per_epoch
                data_epoch = data[i,j,start_indx:end_indx]

                ptp_amp[i,j,k] = mne_features.univariate.compute_ptp_amp(data_epoch)
                variance[i,j,k] = mne_features.univariate.compute_variance(data_epoch)
                rms[i,j,k] = mne_features.univariate.compute_rms(data_epoch)
    
    features = np.stack((ptp_amp, variance, rms))
    features = features.reshape((-1, n_epochs*3))
    return features

def fractal_features(data, new_ica):
    '''
    Compute the Higuchi Fractal Dimension and Katz Fractal Dimension using the package mne_features.
    The data should be on the form (n_trials, n_secs, n_channels, sfreq)
    The output is on the form (n_trials*n_secs, n_channels*2)
    '''
    if new_ica:
        sfreq = v.NEW_SFREQ
    else:
        sfreq = v.SFREQ
    
    n_recordings = data.shape[0]
    n_samples = data.shape[2]
    n_samples_per_epoch = int(n_samples/sfreq)
    n_epochs = int(n_samples/n_samples_per_epoch)
    
    higuchi = np.zeros((n_recordings, v.NUM_CHANNELS, n_epochs))
    katz = np.zeros((n_recordings, v.NUM_CHANNELS, n_epochs))

    for i in range(n_recordings):
        for j in range(n_epochs):
            start_indx = j*n_samples_per_epoch
            end_indx = start_indx + n_samples_per_epoch
            data_epoch = data[i,:,start_indx:end_indx]
            higuchi[i,:,j] = mne_features.univariate.compute_higuchi_fd(data_epoch)
            katz[i,:,j] = mne_features.univariate.compute_katz_fd(data_epoch)
    
    features = np.stack((higuchi, katz), axis = -1)
    n_epochs = features.shape[-2]
    features = features.reshape((-1, n_epochs*2))
    return features

def entropy_features(data, new_ica):
    '''
    Compute the features Approximate Entropy, Sample Entropy, Spectral Entropy and SVD entropy using the package mne_features.
    The data should be on the form (n_trials, n_secs, n_channels, sfreq)
    The output is on the form (n_trials*n_secs, n_channels*4)
    '''
    if new_ica:
        sfreq = v.NEW_SFREQ
    else:
        sfreq = v.SFREQ
    
    n_recordings = data.shape[0]
    n_samples = data.shape[2]
    n_samples_per_epoch = int(n_samples/sfreq)
    n_epochs = int(n_samples/n_samples_per_epoch)
    
    app_entropy = np.zeros((n_recordings, v.NUM_CHANNELS, n_epochs))
    samp_entropy = np.zeros((n_recordings, v.NUM_CHANNELS, n_epochs))
    spect_entropy = np.zeros((n_recordings, v.NUM_CHANNELS, n_epochs))
    svd_entropy = np.zeros((n_recordings, v.NUM_CHANNELS, n_epochs))

    for i in range(n_recordings):
            for j in range(n_epochs):
                start_indx = j*n_samples_per_epoch
                end_indx = start_indx + n_samples_per_epoch
                data_epoch = data[i,:,start_indx:end_indx]

                app_entropy[i,:,j] = mne_features.univariate.compute_app_entropy(data_epoch)
                samp_entropy[i,:,j] = mne_features.univariate.compute_samp_entropy(data_epoch)
                spect_entropy[i,:,j] = mne_features.univariate.compute_spect_entropy(sfreq, data_epoch)
                svd_entropy[i,:,j] = mne_features.univariate.compute_svd_entropy(data_epoch)
        
    features = np.stack((app_entropy, samp_entropy, spect_entropy, svd_entropy), axis = -1)
    n_epochs = features.shape[-2]
    features = features.reshape((-1, n_epochs*4))
    return features

def hjorth_features(data, new_ica):
    '''
    Compute the features Hjorth mobility (spectral) and Hjorth complexity (spectral) using the package mne_features.
    The data should be on the form (n_trials, n_secs, n_channels, sfreq)
    The output is on the form (n_trials*n_secs, n_channels*2)
    '''

    if new_ica:
        sfreq = v.NEW_SFREQ
    else:
        sfreq = v.SFREQ
    
    n_recordings = data.shape[0]
    n_samples = data.shape[2]
    n_samples_per_epoch = int(n_samples/sfreq)
    n_epochs = int(n_samples/n_samples_per_epoch)
    
    mobility_spect = np.zeros((n_recordings, v.NUM_CHANNELS, n_epochs))
    complexity_spect = np.zeros((n_recordings, v.NUM_CHANNELS, n_epochs))

    for i in range(n_recordings):
        for j in range(v.NUM_CHANNELS):
            for k in range(n_epochs):
                start_indx = k*n_samples_per_epoch
                end_indx = start_indx + n_samples_per_epoch
                data_epoch = data[i,j,start_indx:end_indx]

                mobility_spect[i,:,j] = mne_features.univariate.compute_hjorth_mobility_spect(sfreq, data_epoch)
                complexity_spect[i,:,j] = mne_features.univariate.compute_hjorth_complexity_spect(sfreq, data_epoch)
        
    features = np.stack((mobility_spect, complexity_spect), axis = -1)
    n_epochs = features.shape[-2]
    features = features.reshape((-1, n_epochs*2))
    return features

def freq_band_features(data, new_ica, freq_bands = np.array([1, 4, 8, 12, 30, 50])):
    '''
    Compute the frequency bands delta, theta, alpha, beta and gamma using the package mne_features.
    The data should be on the form (n_trials, n_secs, n_channels, sfreq)
    The output is on the form (n_trials*n_secs, n_channels*5)
    '''
    if new_ica:
        sfreq = v.NEW_SFREQ
    else:
        sfreq = v.SFREQ
    
    n_recordings = data.shape[0]
    n_samples = data.shape[2]
    n_samples_per_epoch = int(n_samples/sfreq)
    n_epochs = int(n_samples/n_samples_per_epoch)
    
    psd = np.zeros((n_recordings, v.NUM_CHANNELS*(len(freq_bands)-1), n_epochs))

    for i in range(n_recordings):
        for j in range(n_epochs):
            start_indx = j*n_samples_per_epoch
            end_indx = start_indx + n_samples_per_epoch
            data_epoch = data[i,:,start_indx:end_indx]

            psd[i,:,j] =  mne_features.univariate.compute_pow_freq_bands(
            v.SFREQ, data_epoch, freq_bands=freq_bands, normalize=True, ratios=None, ratios_triu=False, psd_method='welch', log=False, psd_params=None)


    features = np.stack((psd), axis = -1)
    n_epochs = features.shape[-2]
    features = features.reshape((-1, n_epochs*(len(freq_bands)-1)))
    return features


    '''
    Computes all features included in the package mne_features.
    Args:
        data (list of dicts): A list of dictionaries, where each dictionary contains EEG data for multiple trials. The keys in each dictionary represent trial IDs, and the values are numpy arrays of shape (n_channels, n_samples).
    Returns:
        list of ndarrays: Computed features.
    '''
    
    first_key = next(iter(data[0]))
    n_channels, _ = data[0][first_key].shape
    features_per_channel = 10
    

    features = []
    for fold in data:
        n_trials = len(fold)
        features_for_fold = np.empty([n_trials, n_channels * features_per_channel])
        for j, key in enumerate(fold):
            trial = fold[key]
            std = mne_f.compute_std(trial)
            ptp_amp = mne_f.compute_ptp_amp(trial)
            skewness = mne_f.compute_skewness(trial)
            kurtosis = mne_f.compute_kurtosis(trial)
            rms = mne_f.compute_rms(trial)
            quantile = mne_f.compute_quantile(trial)
            hurst = mne_f.compute_hurst_exp(trial)
            app_entropy = mne_f.compute_app_entropy(trial)
            mobility_spect = mne_f.compute_hjorth_mobility_spect(v.SFREQ, trial)
            complexity_spect = mne_f.compute_hjorth_complexity_spect(v.SFREQ, trial)
            mobility = mne_f.compute_hjorth_mobility(trial)
            complexity = mne_f.compute_hjorth_complexity(trial)

            features_for_fold[j] =  np.concatenate([std, ptp_amp, skewness, kurtosis, rms, quantile, hurst, app_entropy, mobility_spect, complexity_spect, mobility, complexity])
                        
        features_for_fold = features_for_fold.reshape([n_trials, n_channels*features_per_channel])
        features.append(features_for_fold)
    return features