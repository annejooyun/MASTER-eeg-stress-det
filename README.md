# EEG Stress Detection
Classification of stress using EEG recordings.


## Dataset
All data is available in \Data folder.
Here, different versions of the data can be found.
- Raw_eeg, which includes the untouched data
- Init_filter_data, which includes the data after initial filtering (band-pass and Savitsky-Golay filtering)
- ICA_data, which includes the data after artifact removal with Independent Component Analysis


## Files
The code is split into Jupyter notebooks and Python scripts.

### utils
Folder with all "help-functions"

**variables.py**
Includes all important variables

**data.py**
Includes functions for loading eeg data, switching the dataset from multi to binary classification, splitting data into train-, validation- and test-sets etc.

**labels.py**
Includes functions for computing stress labels, either with PSS or STAI-Y

**valid_recs.py**
Includes functions for filtering out invalid recordings

**metrics.py**
Includes a function that computes accuracy, spesificity and sensitivity of a classification

### extract_eeg.ipynb
Jupyter Notebook for loading .xdf-file, extracting EEG data, markers and PCG data, and saving the EEG data in .mat format

### channel_selection.ipynb
Jupyter Notebook for channel selection using the Genetic Algorithm.
The script uses the best performing settings to reduce the number of electrodes from 32 to 8 with as little loss in performance as possible.

### filtering.ipynb
Jupyter Notebook for filtering EEG data.
Raw EEG is filtered using a band-pass and a Savitsky-Golay filter.Then, artifact removal is performed usin an Independent Component Analysis.
The filtering is performed using the ```mne``` package which is a Python package specialised in MEG and EEG analysis and visualisation.

### kfold_classification.ipynb
Jupyter Notebook for classification using k-fold split and KNN, SVM or MLP

### EEGNet_classification.ipynb
Jupyter Notebook for classification splitting along subjects using EEGNet's implementations of various deep neural networks.

### features.py
Includes functions to compute time series, nonlinear, entropy, hjorth and frequency band features. Used before feeding the data into KNN, SVM or MLP.

### EEGModels.py
ARL_EEGModels - A collection of Convolutional Neural Network models for EEG Signal Processing and Classification, using Keras and Tensorflow. Repurposed from 
https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py

### genetic_alg.py
Includes support functions to the Genetic Algorithm. Repurposed from https://github.com/ahmedfgad/GeneticAlgorithmPython



