
import mne_features.univariate as mne_f
import numpy as np
import utils.variables as v


def time_series_features(data):
    '''
    Computes the features variance, RMS and peak-to-peak amplitude using the package mne_features.
    Args:
        data (list of dicts): A list of dictionaries, where each dictionary contains EEG data for multiple trials. The keys in each dictionary represent trial IDs, and the values are numpy arrays of shape (n_channels, n_samples).
    Returns:
        list of ndarrays: Computed features.
    '''
    first_key = next(iter(data[0]))
    n_channels, _ = data[0][first_key].shape
    features_per_channel = 3

    features = []
    for fold in data:
        n_trials = len(fold)
        features_for_fold = np.empty(
            [n_trials, n_channels * features_per_channel])
        for j, key in enumerate(fold):
            trial = fold[key]
            variance = mne_f.compute_variance(trial)
            rms = mne_f.compute_rms(trial)
            ptp_amp = mne_f.compute_ptp_amp(trial)
            features_for_fold[j] = np.concatenate([variance, rms, ptp_amp])
        features_for_fold = features_for_fold.reshape(
            [n_trials, n_channels*features_per_channel])
        features.append(features_for_fold)
    return features


def freq_band_features(data, freq_bands):
    '''
    Computes the frequency bands delta, theta, alpha, beta and gamma using the package mne_features.
    Args:
        data (list of dicts): A list of dictionaries, where each dictionary contains EEG data for multiple trials. The keys in each dictionary represent trial IDs, and the values are numpy arrays of shape (n_channels, n_samples).
        freq_bands (ndarray): The frequency bands to compute.
    Returns:
        list of ndarrays: Computed features.
    '''
    first_key = next(iter(data[0]))
    n_channels, _ = data[0][first_key].shape
    features_per_channel = len(freq_bands)-1

    features = []
    for fold in data:
        n_trials = len(fold)
        features_for_fold = np.empty(
            [n_trials, n_channels * features_per_channel])
        for j, key in enumerate(fold):
            trial = fold[key]
            psd = mne_f.compute_pow_freq_bands(
                v.SFREQ, trial, freq_bands=freq_bands)
            features_for_fold[j] = psd
        features_for_fold = features_for_fold.reshape(
            [n_trials, n_channels*features_per_channel])
        features.append(features_for_fold)
    return features


def hjorth_features(data):
    '''
    Computes the features Hjorth mobility (spectral) and Hjorth complexity (spectral) using the package mne_features.
    Args:
        data (list of dicts): A list of dictionaries, where each dictionary contains EEG data for multiple trials. The keys in each dictionary represent trial IDs, and the values are numpy arrays of shape (n_channels, n_samples).
    Returns:
        list of ndarrays: Computed features.
    '''

    first_key = next(iter(data[0]))
    n_channels, _ = data[0][first_key].shape
    features_per_channel = 2

    features = []
    for fold in data:
        n_trials = len(fold)
        features_for_fold = np.empty(
            [n_trials, n_channels * features_per_channel])
        for j, key in enumerate(fold):
            trial = fold[key]
            mobility_spect = mne_f.compute_hjorth_mobility_spect(v.SFREQ, trial)
            complexity_spect = mne_f.compute_hjorth_complexity_spect(v.SFREQ, trial)
            features_for_fold[j] = np.concatenate([mobility_spect, complexity_spect])
        features_for_fold = features_for_fold.reshape([n_trials, n_channels*features_per_channel])
        features.append(features_for_fold)
    return features


def fractal_features(data):
    '''
    Computes the Higuchi Fractal Dimension and Katz Fractal Dimension using the package mne_features.
    Args:
        data (ndarray): EEG data.
    Returns:
        list of ndarrays: Computed features.
    '''

    first_key = next(iter(data[0]))
    n_channels, _ = data[0][first_key].shape
    features_per_channel = 2

    features = []
    for fold in data:
        n_trials = len(fold)
        features_for_fold = np.empty(
            [n_trials, n_channels * features_per_channel])
        for j, key in enumerate(fold):
            trial = fold[key]
            higuchi = mne_f.compute_higuchi_fd(trial)
            katz = mne_f.compute_katz_fd(trial)
            features_for_fold[j] = np.concatenate([higuchi, katz])
        features_for_fold = features_for_fold.reshape(
            [n_trials, n_channels*features_per_channel])
        features.append(features_for_fold)
    return features


def entropy_features(data):
    '''
    Computes the features Approximate Entropy, Sample Entropy, Spectral Entropy and SVD entropy using the package mne_features.
    Args:
        data (list of dicts): A list of dictionaries, where each dictionary contains EEG data for multiple trials. The keys in each dictionary represent trial IDs, and the values are numpy arrays of shape (n_channels, n_samples).
    Returns:
        list of ndarrays: Computed features.
    '''

    first_key = next(iter(data[0]))
    n_channels, _ = data[0][first_key].shape
    features_per_channel = 4

    features = []
    for fold in data:
        n_trials = len(fold)
        features_for_fold = np.empty(
            [n_trials, n_channels * features_per_channel])
        for j, key in enumerate(fold):
            trial = fold[key]
            app_entropy = mne_f.compute_app_entropy(trial)
            samp_entropy = mne_f.compute_samp_entropy(trial)
            spect_entropy = mne_f.compute_spect_entropy(v.SFREQ, trial)
            svd_entropy = mne_f.compute_svd_entropy(trial)
            features_for_fold[j] = np.concatenate(
                [app_entropy, samp_entropy, spect_entropy, svd_entropy])
        features_for_fold = features_for_fold.reshape(
            [n_trials, n_channels*features_per_channel])
        features.append(features_for_fold)
    return 


def all_features_1(data):
    '''
    Computes all features included in the package mne_features.
    Args:
        data (list of dicts): A list of dictionaries, where each dictionary contains EEG data for multiple trials. The keys in each dictionary represent trial IDs, and the values are numpy arrays of shape (n_channels, n_samples).
    Returns:
        list of ndarrays: Computed features.
    '''
    
    first_key = next(iter(data[0]))
    n_channels, _ = data[0][first_key].shape
    features_per_channel = 24
    

    features = []
    for fold in data:
        n_trials = len(fold)
        features_for_fold = np.empty([n_trials, n_channels * features_per_channel])
        for j, key in enumerate(fold):
            trial = fold[key]
            mean = mne_f.compute_mean(trial)
            variance = mne_f.compute_variance(trial)
            std = mne_f.compute_std(trial)
            ptp_amp = mne_f.compute_ptp_amp(trial)
            skewness = mne_f.compute_skewness(trial)
            kurtosis = mne_f.compute_kurtosis(trial)
            rms = mne_f.compute_rms(trial)
            quantile = mne_f.compute_quantile(trial)
            hurst = mne_f.compute_hurst_exp(trial)
            app_entropy = mne_f.compute_app_entropy(trial)
            samp_entropy = mne_f.compute_samp_entropy(trial)
            decorr_time = mne_f.compute_decorr_time(v.SFREQ, trial)
            mobility_spect = mne_f.compute_hjorth_mobility_spect(v.SFREQ, trial)
            complexity_spect = mne_f.compute_hjorth_complexity_spect(v.SFREQ, trial)
            mobility = mne_f.compute_hjorth_mobility(trial)
            complexity = mne_f.compute_hjorth_complexity(trial)
            h_fd = mne_f.compute_higuchi_fd(trial)
            k_fd = mne_f.compute_katz_fd(trial)
            zero_crossings = mne_f.compute_zero_crossings(trial)
            line_length = mne_f.compute_line_length(trial)
            spect_entropy = mne_f.compute_spect_entropy(v.SFREQ, trial)
            svd_entropy = mne_f.compute_svd_entropy(trial)
            fisher = mne_f.compute_svd_fisher_info(trial)
            spect_edge_freq = mne_f.compute_spect_edge_freq(v.SFREQ, trial)

            features_for_fold[j] =  np.concatenate([mean, variance, std, ptp_amp, skewness, kurtosis, rms, quantile, hurst, app_entropy, samp_entropy, 
                                                    decorr_time, mobility_spect, complexity_spect, mobility, complexity, h_fd, k_fd, zero_crossings, 
                                                    line_length, spect_entropy, svd_entropy, fisher, spect_edge_freq])
                        
        features_for_fold = features_for_fold.reshape([n_trials, n_channels*features_per_channel])
        features.append(features_for_fold)
    return features


def all_features(data):
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