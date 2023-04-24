import numpy as np
from utils.metrics import compute_metrics

from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from keras import models, Input
from keras import optimizers as opt
from keras import backend as K
from keras.layers import Dense
#from keras_tuner.tuners import RandomSearch
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt

# EEGNet-specific imports
from EEGModels import EEGNet,EEGNet_SSVEP,TSGLEEGNet, DeepConvNet, ShallowConvNet, TSGLEEGNet
from tensorflow.keras.callbacks import ModelCheckpoint

import utils.variables as v



def knn_classification(x_train, x_test, y_train, y_test):
    n_splits = len(x_train)

    param_grid = {
        'leaf_size': range(50),
        'n_neighbors': range(1, 10),
        'p': [1, 2]
    }

    results = []
    for i in range(n_splits):
        x_train_fold = x_train[i]
        x_test_fold = x_test[i]

        y_train_fold = y_train[i]
        y_test_fold = y_test[i]

        scaler = MinMaxScaler()
        x_train_fold = scaler.fit_transform(x_train_fold)
        x_test_fold = scaler.transform(x_test_fold)

        knn_clf = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, n_jobs=-1)
        knn_clf.fit(x_train_fold, y_train_fold)

        y_pred = knn_clf.predict(x_test_fold)
        y_true = y_test_fold

        print(f'\nResults for fold {i+1}:')
        print(metrics.classification_report(y_true, y_pred))
        print(metrics.confusion_matrix(y_true, y_pred))
        



def svm_classification(x_train, x_test, y_train, y_test):
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'kernel': ['rbf']
    }

    n_splits = len(x_train)

    for i in range(n_splits):
        x_train_fold = x_train[i]
        x_test_fold = x_test[i]

        y_train_fold = y_train[i]
        y_test_fold = y_test[i]

        scaler = MinMaxScaler()
        scaler.fit(x_train_fold)
        x_train_fold = scaler.transform(x_train_fold)
        x_test_fold = scaler.transform(x_test_fold)

        svm_clf = GridSearchCV(SVC(), param_grid, refit=True)
        svm_clf.fit(x_train_fold, y_train_fold)

        y_pred = svm_clf.predict(x_test_fold)
        y_true = y_test_fold

        print(f'\nResults for fold {i+1}:')
        print(metrics.classification_report(y_true, y_pred))
        print(metrics.confusion_matrix(y_true, y_pred))





def nn_classification(data, label):
    K.clear_session()
    y_v = label
    y_v = to_categorical(y_v)
    x_train, x_test, y_train, y_test = train_test_split(
        data, y_v, test_size=0.2, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.25, random_state=1)

    def model_builder(hp):
        model = models.Sequential()
        model.add(Input(shape=(x_train.shape[1],)))

        for i in range(hp.Int('layers', 2, 6)):
            model.add(Dense(units=hp.Int('units_' + str(i), 32, 1024, step=32),
                            activation=hp.Choice('act_' + str(i), ['relu', 'sigmoid'])))

        model.add(Dense(v.N_CLASSES, activation='softmax', name='out'))

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        model.compile(optimizer=opt.adam_v2.Adam(learning_rate=hp_learning_rate),
                    loss="binary_crossentropy",
                    metrics=['accuracy'])
        return model


    tuner = RandomSearch(
        model_builder,
        objective='val_accuracy',
        max_trials=15,
        executions_per_trial=2,
        overwrite=True
    )

    tuner.search_space_summary()

    tuner.search(x_train, y_train, epochs=50, validation_data=[x_val, y_val])

    model = tuner.get_best_models(num_models=1)[0]

    y_pred = model.predict(x_test)
    y_true = y_test
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)

    scores_dnn = model.evaluate(x_test, y_test, verbose=0)

    print(metrics.classification_report(y_true, y_pred))
    print(metrics.confusion_matrix(y_true, y_pred))



def EEGNet_classification(train_data, test_data, val_data, train_labels, test_labels, val_labels, data_type, epoched = True):

    # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other 
    # model configurations may do better, but this is a good starting point)
    if epoched:
        if data_type == 'new_ica':
            model = EEGNet(nb_classes = 2, Chans = v.NUM_CHANNELS , Samples = v.EPOCH_LENGTH*v.NEW_SFREQ, 
                    dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
                    dropoutType = 'Dropout')
        else:
            model = EEGNet(nb_classes = 2, Chans = v.NUM_CHANNELS , Samples = v.EPOCH_LENGTH*v.SFREQ, 
                    dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
                    dropoutType = 'Dropout')
    else: #if not epoched
        if data_type == 'new_ica':
            model = EEGNet(nb_classes = 2, Chans = v.NUM_CHANNELS , Samples = v.NEW_NUM_SAMPLES, 
                    dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
                    dropoutType = 'Dropout')
        else:
            model = EEGNet(nb_classes = 2, Chans = v.NUM_CHANNELS , Samples = v.NUM_SAMPLES, 
                    dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
                    dropoutType = 'Dropout')


    # compile the model and set the optimizers
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
                metrics = ['accuracy'])

    # count number of parameters in the model
    numParams    = model.count_params()    

    # set a valid path for your system to record model checkpoints
    checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                                save_best_only=True)

    ###############################################################################
    # if the classification task was imbalanced (significantly more trials in one
    # class versus the others) you can assign a weight to each class during 
    # optimization to balance it out. This data is approximately balanced so we 
    # don't need to do this, but is shown here for illustration/completeness. 
    ###############################################################################

    # the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
    # the weights all to be 1
    class_weights = {0:1, 1:3}

    ################################################################################
    # fit the model. Due to very small sample sizes this can get
    # pretty noisy run-to-run, but most runs should be comparable to xDAWN + 
    # Riemannian geometry classification (below)
    ################################################################################
    '''fittedModel = model.fit(train_data, train_labels, batch_size = 64, epochs = 300, 
                            verbose = 2, validation_data=(val_data, val_labels),
                            callbacks=[checkpointer], class_weight = class_weights)'''
    history = model.fit(train_data, train_labels, batch_size = 32, epochs = 200, 
                            verbose = 2, validation_data=(val_data, val_labels),
                            callbacks=[checkpointer], class_weight = class_weights)

    # load optimal weights
    model.load_weights('/tmp/checkpoint.h5')

    ###############################################################################
    # can alternatively used the weights provided in the repo. If so it should get
    # you 93% accuracy. Change the WEIGHTS_PATH variable to wherever it is on your
    # system.
    ###############################################################################

    # WEIGHTS_PATH = /path/to/EEGNet-8-2-weights.h5 
    # model.load_weights(WEIGHTS_PATH)

    ###############################################################################
    # make prediction on test set.
    ###############################################################################

    probs       = model.predict(test_data)
    preds       = probs.argmax(axis = -1)  
    acc         = np.mean(preds == test_labels)
    print("Classification accuracy: %f " % (acc))

    # print performance
    performance = compute_metrics(test_labels, preds)
    print("Accuracy, Sensitivity, Specificyty:\n")
    print(performance)

    
    # Plot Loss/Accuracy over time
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(go.Scatter( y=history.history['val_loss'], name="val_loss"), secondary_y=False)
    fig.add_trace(go.Scatter( y=history.history['loss'], name="loss"), secondary_y=False)
    fig.add_trace(go.Scatter( y=history.history['val_accuracy'], name="val accuracy"), secondary_y=True)
    fig.add_trace(go.Scatter( y=history.history['accuracy'], name="accuracy"), secondary_y=True)
    # Add figure title
    fig.update_layout(title_text="Loss/Accuracy of EEGNet")
    # Set x-axis title
    fig.update_xaxes(title_text="Epoch")
    # Set y-axes titles
    fig.update_yaxes(title_text="Loss", secondary_y=False)
    fig.update_yaxes(title_text="Accuracy", secondary_y=True)
    fig.show()


    # plot the confusion matrices for both classifiers
    conf_matrix = metrics.confusion_matrix(test_labels,preds)
    print(conf_matrix)
    return probs

def kfold_EEGNet_classification(train_data, test_data, train_labels, test_labels, n_folds, data_type, epoched = True):

    if epoched:
        if data_type == 'new_ica':
            model = EEGNet(nb_classes = 2, Chans = v.NUM_CHANNELS , Samples = v.EPOCH_LENGTH*v.NEW_SFREQ, 
                    dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
                    dropoutType = 'Dropout')
        else:
            model = EEGNet(nb_classes = 2, Chans = v.NUM_CHANNELS , Samples = v.EPOCH_LENGTH*v.SFREQ, 
                    dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
                    dropoutType = 'Dropout')
    else: #if not epoched
        if data_type == 'new_ica':
            model = EEGNet(nb_classes = 2, Chans = v.NUM_CHANNELS , Samples = v.NEW_NUM_SAMPLES, 
                    dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
                    dropoutType = 'Dropout')
        else:
            model = EEGNet(nb_classes = 2, Chans = v.NUM_CHANNELS , Samples = v.NUM_SAMPLES, 
                    dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
                    dropoutType = 'Dropout')

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
                metrics = ['accuracy'])

    numParams    = model.count_params()    

    checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                                save_best_only=True)

    class_weights = {0:1, 1:3}

    # Split into k-folds
    skf = StratifiedKFold(n_splits=n_folds)
    total_accuracy = 0

    for fold, (train_index, val_index) in enumerate(skf.split(train_data, train_labels)):
        print(f"\nFold nr: {fold+1}")
        train_data_fold, train_labels_fold = train_data[train_index], train_labels[train_index]
        val_data_fold, val_labels_fold = train_data[val_index], train_labels[val_index]

        history = model.fit(train_data_fold, train_labels_fold, batch_size = 16, epochs = 100, 
                            verbose = 2, validation_data=(val_data_fold, val_labels_fold),
                            callbacks=[checkpointer], class_weight = class_weights)

        # load optimal weights
        model.load_weights('/tmp/checkpoint.h5')

        probs       = model.predict(test_data)
        preds       = probs.argmax(axis = -1)  
        acc         = np.mean(preds == test_labels)
        total_accuracy += acc
        print("Classification accuracy: %f " % (acc))

        print(classification_report(test_labels, preds))
        
        # Plot Loss/Accuracy over time
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Add traces
        fig.add_trace(go.Scatter( y=history.history['val_loss'], name="val_loss"), secondary_y=False)
        fig.add_trace(go.Scatter( y=history.history['loss'], name="loss"), secondary_y=False)
        fig.add_trace(go.Scatter( y=history.history['val_accuracy'], name="val accuracy"), secondary_y=True)
        fig.add_trace(go.Scatter( y=history.history['accuracy'], name="accuracy"), secondary_y=True)
        # Add figure title
        fig.update_layout(title_text="Loss/Accuracy of k-folds EEGNet")
        # Set x-axis title
        fig.update_xaxes(title_text="Epoch")
        # Set y-axes titles
        fig.update_yaxes(title_text="Loss", secondary_y=False)
        fig.update_yaxes(title_text="Accuracy", secondary_y=True)
        fig.show()
    
    classification_acc = total_accuracy/n_folds
    print(f"EEGNet overall classification accuracy: {classification_acc}")


def TSGL_classification(train_data, test_data, val_data, train_labels, test_labels, val_labels, data_type, epoched = True):

    if epoched:
        if data_type == 'new_ica':
            model = TSGLEEGNet(nb_classes = 2, Chans = v.NUM_CHANNELS, Samples = v.EPOCH_LENGTH*v.NEW_SFREQ, 
                                dropoutRate = 0.5, kernLength = 128, F1 = 96, D = 1, F2 = 96, dropoutType = 'Dropout')
        else:
            model = TSGLEEGNet(nb_classes = 2, Chans = v.NUM_CHANNELS, Samples = v.EPOCH_LENGTH*v.SFREQ, 
                                dropoutRate = 0.5, kernLength = 128, F1 = 96, D = 1, F2 = 96, dropoutType = 'Dropout')
    else: #if not epoched
        if data_type == 'new_ica':
           model = TSGLEEGNet(nb_classes = 2, Chans = v.NUM_CHANNELS, Samples = v.NEW_NUM_SAMPLES, 
                                dropoutRate = 0.5, kernLength = 128, F1 = 96, D = 1, F2 = 96, dropoutType = 'Dropout')
        else:
            model = TSGLEEGNet(nb_classes = 2, Chans = v.NUM_CHANNELS, Samples = v.NUM_SAMPLES, 
                                 dropoutRate = 0.5, kernLength = 128, F1 = 96, D = 1, F2 = 96, dropoutType = 'Dropout')
            
    # compile the model and set the optimizers
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
                metrics = ['accuracy'])

    # count number of parameters in the model
    numParams    = model.count_params()    

    # set a valid path for your system to record model checkpoints
    checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                                save_best_only=True)

    class_weights = {0:1, 1:3}

    # fit the model
    history = model.fit(train_data, train_labels, batch_size = 32, epochs = 200, 
                            verbose = 2, validation_data=(val_data, val_labels),
                            callbacks=[checkpointer], class_weight = class_weights)

    # load optimal weights
    model.load_weights('/tmp/checkpoint.h5')

    # make prediction on test set.
    probs       = model.predict(test_data)
    preds       = probs.argmax(axis = -1)  
    acc         = np.mean(preds == test_labels)
    print("Classification accuracy: %f " % (acc))

    # print performance
    performance = compute_metrics(test_labels, preds)
    print("Accuracy, Sensitivity, Specificyty:\n")
    print(performance)

    
    # Plot Loss/Accuracy over time
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(go.Scatter( y=history.history['val_loss'], name="val_loss"), secondary_y=False)
    fig.add_trace(go.Scatter( y=history.history['loss'], name="loss"), secondary_y=False)
    fig.add_trace(go.Scatter( y=history.history['val_accuracy'], name="val accuracy"), secondary_y=True)
    fig.add_trace(go.Scatter( y=history.history['accuracy'], name="accuracy"), secondary_y=True)
    # Add figure title
    fig.update_layout(title_text="Loss/Accuracy of TSGL")
    # Set x-axis title
    fig.update_xaxes(title_text="Epoch")
    # Set y-axes titles
    fig.update_yaxes(title_text="Loss", secondary_y=False)
    fig.update_yaxes(title_text="Accuracy", secondary_y=True)
    fig.show()


    # plot the confusion matrices for both classifiers
    conf_matrix = metrics.confusion_matrix(test_labels,preds)
    print(conf_matrix)
    return probs

def kfold_TSGL_classification(train_data, test_data, train_labels, test_labels, n_folds, data_type, epoched = True):

    if epoched:
        if data_type == 'new_ica':
            model = TSGLEEGNet(nb_classes = 2, Chans = v.NUM_CHANNELS, Samples = v.EPOCH_LENGTH*v.NEW_SFREQ, 
                                dropoutRate = 0.5, kernLength = 128, F1 = 96, D = 1, F2 = 96, dropoutType = 'Dropout')
        else:
            model = TSGLEEGNet(nb_classes = 2, Chans = v.NUM_CHANNELS, Samples = v.EPOCH_LENGTH*v.SFREQ, 
                                dropoutRate = 0.5, kernLength = 128, F1 = 96, D = 1, F2 = 96, dropoutType = 'Dropout')
    else: #if not epoched
        if data_type == 'new_ica':
           model = TSGLEEGNet(nb_classes = 2, Chans = v.NUM_CHANNELS, Samples = v.NEW_NUM_SAMPLES, 
                                dropoutRate = 0.5, kernLength = 128, F1 = 96, D = 1, F2 = 96, dropoutType = 'Dropout')
        else:
            model = TSGLEEGNet(nb_classes = 2, Chans = v.NUM_CHANNELS, Samples = v.NUM_SAMPLES, 
                                 dropoutRate = 0.5, kernLength = 128, F1 = 96, D = 1, F2 = 96, dropoutType = 'Dropout')
            
    # compile the model and set the optimizers
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
                metrics = ['accuracy'])

    # count number of parameters in the model
    numParams    = model.count_params()    

    # set a valid path for your system to record model checkpoints
    checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                                save_best_only=True)

    class_weights = {0:1, 1:3}

     # Split into k-folds
    skf = StratifiedKFold(n_splits=n_folds)
    total_accuracy = 0

    for fold, (train_index, val_index) in enumerate(skf.split(train_data, train_labels)):
        print(f"\nFold nr: {fold+1}")
        train_data_fold, train_labels_fold = train_data[train_index], train_labels[train_index]
        val_data_fold, val_labels_fold = train_data[val_index], train_labels[val_index]

        history = model.fit(train_data_fold, train_labels_fold, batch_size = 16, epochs = 100, 
                            verbose = 2, validation_data=(val_data_fold, val_labels_fold),
                            callbacks=[checkpointer], class_weight = class_weights)

        # load optimal weights
        model.load_weights('/tmp/checkpoint.h5')

        probs       = model.predict(test_data)
        preds       = probs.argmax(axis = -1)  
        acc         = np.mean(preds == test_labels)
        total_accuracy += acc
        print("Classification accuracy: %f " % (acc))

        print(classification_report(test_labels, preds))
        
        # Plot Loss/Accuracy over time
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Add traces
        fig.add_trace(go.Scatter( y=history.history['val_loss'], name="val_loss"), secondary_y=False)
        fig.add_trace(go.Scatter( y=history.history['loss'], name="loss"), secondary_y=False)
        fig.add_trace(go.Scatter( y=history.history['val_accuracy'], name="val accuracy"), secondary_y=True)
        fig.add_trace(go.Scatter( y=history.history['accuracy'], name="accuracy"), secondary_y=True)
        # Add figure title
        fig.update_layout(title_text="Loss/Accuracy of k-fold TSGLs")
        # Set x-axis title
        fig.update_xaxes(title_text="Epoch")
        # Set y-axes titles
        fig.update_yaxes(title_text="Loss", secondary_y=False)
        fig.update_yaxes(title_text="Accuracy", secondary_y=True)
        fig.show()
    
    classification_acc = total_accuracy/n_folds
    print(f"TSGL overall classification accuracy: {classification_acc}")


def DeepConvNet_classification(train_data, test_data, val_data, train_labels, test_labels, val_labels, data_type, epoched = True):
    if epoched:
        if data_type == 'new_ica':
            model = DeepConvNet(nb_classes = 2, Chans = v.NUM_CHANNELS, Samples = v.EPOCH_LENGTH*v.NEW_SFREQ, 
                                dropoutRate = 0.5)
        else:
            model = DeepConvNet(nb_classes = 2, Chans = v.NUM_CHANNELS, Samples = v.EPOCH_LENGTH*v.SFREQ, 
                                dropoutRate = 0.5)
    else: #if not epoched
        if data_type == 'new_ica':
           model = DeepConvNet(nb_classes = 2, Chans = v.NUM_CHANNELS, Samples = v.NEW_NUM_SAMPLES, 
                                dropoutRate = 0.5)
        else:
            model = DeepConvNet(nb_classes = 2, Chans = v.NUM_CHANNELS, Samples = v.NUM_SAMPLES, 
                                 dropoutRate = 0.5)
            
    # compile the model and set the optimizers
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
                metrics = ['accuracy'])

    # count number of parameters in the model
    numParams    = model.count_params()    

    # set a valid path for your system to record model checkpoints
    checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                                save_best_only=True)

    class_weights = {0:1, 1:3}

    # fit the model
    history = model.fit(train_data, train_labels, batch_size = 32, epochs = 200, 
                            verbose = 2, validation_data=(val_data, val_labels),
                            callbacks=[checkpointer], class_weight = class_weights)

    # load optimal weights
    model.load_weights('/tmp/checkpoint.h5')

    # make prediction on test set.
    probs       = model.predict(test_data)
    preds       = probs.argmax(axis = -1)  
    acc         = np.mean(preds == test_labels)
    print("Classification accuracy: %f " % (acc))

    # print performance
    performance = compute_metrics(test_labels, preds)
    print("Accuracy, Sensitivity, Specificyty:\n")
    print(performance)

    
    # Plot Loss/Accuracy over time
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(go.Scatter( y=history.history['val_loss'], name="val_loss"), secondary_y=False)
    fig.add_trace(go.Scatter( y=history.history['loss'], name="loss"), secondary_y=False)
    fig.add_trace(go.Scatter( y=history.history['val_accuracy'], name="val accuracy"), secondary_y=True)
    fig.add_trace(go.Scatter( y=history.history['accuracy'], name="accuracy"), secondary_y=True)
    # Add figure title
    fig.update_layout(title_text="Loss/Accuracy of DeepConvNet")
    # Set x-axis title
    fig.update_xaxes(title_text="Epoch")
    # Set y-axes titles
    fig.update_yaxes(title_text="Loss", secondary_y=False)
    fig.update_yaxes(title_text="Accuracy", secondary_y=True)
    fig.show()


    # plot the confusion matrices for both classifiers
    conf_matrix = metrics.confusion_matrix(test_labels,preds)
    print(conf_matrix)
    return probs

def kfold_DeepConvNet_classification(train_data, test_data, train_labels, test_labels, n_folds, data_type, epoched = True):
    if epoched:
        if data_type == 'new_ica':
            model = DeepConvNet(nb_classes = 2, Chans = v.NUM_CHANNELS, Samples = v.EPOCH_LENGTH*v.NEW_SFREQ, 
                                dropoutRate = 0.5)
        else:
            model = DeepConvNet(nb_classes = 2, Chans = v.NUM_CHANNELS, Samples = v.EPOCH_LENGTH*v.SFREQ, 
                                dropoutRate = 0.5)
    else: #if not epoched
        if data_type == 'new_ica':
           model = DeepConvNet(nb_classes = 2, Chans = v.NUM_CHANNELS, Samples = v.NEW_NUM_SAMPLES, 
                                dropoutRate = 0.5)
        else:
            model = DeepConvNet(nb_classes = 2, Chans = v.NUM_CHANNELS, Samples = v.NUM_SAMPLES, 
                                 dropoutRate = 0.5)
            
    # compile the model and set the optimizers
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
                metrics = ['accuracy'])

    # count number of parameters in the model
    numParams    = model.count_params()    

    # set a valid path for your system to record model checkpoints
    checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                                save_best_only=True)

    class_weights = {0:1, 1:3}

     # Split into k-folds
    skf = StratifiedKFold(n_splits=n_folds)
    total_accuracy = 0

    for fold, (train_index, val_index) in enumerate(skf.split(train_data, train_labels)):
        print(f"\nFold nr: {fold+1}")
        train_data_fold, train_labels_fold = train_data[train_index], train_labels[train_index]
        val_data_fold, val_labels_fold = train_data[val_index], train_labels[val_index]

        history = model.fit(train_data_fold, train_labels_fold, batch_size = 16, epochs = 100, 
                            verbose = 2, validation_data=(val_data_fold, val_labels_fold),
                            callbacks=[checkpointer], class_weight = class_weights)

        # load optimal weights
        model.load_weights('/tmp/checkpoint.h5')

        probs       = model.predict(test_data)
        preds       = probs.argmax(axis = -1)  
        acc         = np.mean(preds == test_labels)
        total_accuracy += acc
        print("Classification accuracy: %f " % (acc))

        print(classification_report(test_labels, preds))
        
        # Plot Loss/Accuracy over time
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Add traces
        fig.add_trace(go.Scatter( y=history.history['val_loss'], name="val_loss"), secondary_y=False)
        fig.add_trace(go.Scatter( y=history.history['loss'], name="loss"), secondary_y=False)
        fig.add_trace(go.Scatter( y=history.history['val_accuracy'], name="val accuracy"), secondary_y=True)
        fig.add_trace(go.Scatter( y=history.history['accuracy'], name="accuracy"), secondary_y=True)
        # Add figure title
        fig.update_layout(title_text="Loss/Accuracy of kfold DeepConvNet")
        # Set x-axis title
        fig.update_xaxes(title_text="Epoch")
        # Set y-axes titles
        fig.update_yaxes(title_text="Loss", secondary_y=False)
        fig.update_yaxes(title_text="Accuracy", secondary_y=True)
        fig.show()
    
    classification_acc = total_accuracy/n_folds
    print(f"Deep overall classification accuracy: {classification_acc}")


def ShallowConvNet_classification(train_data, test_data, val_data, train_labels, test_labels, val_labels, data_type, epoched = True):

    if epoched:
        if data_type == 'new_ica':
            model = ShallowConvNet(nb_classes = 2, Chans = v.NUM_CHANNELS, Samples = v.EPOCH_LENGTH*v.NEW_SFREQ, 
                                dropoutRate = 0.5)
        else:
            model = ShallowConvNet(nb_classes = 2, Chans = v.NUM_CHANNELS, Samples = v.EPOCH_LENGTH*v.SFREQ, 
                                dropoutRate = 0.5)
    else: #if not epoched
        if data_type == 'new_ica':
           model = ShallowConvNet(nb_classes = 2, Chans = v.NUM_CHANNELS, Samples = v.NEW_NUM_SAMPLES, 
                                dropoutRate = 0.5)
        else:
            model = ShallowConvNet(nb_classes = 2, Chans = v.NUM_CHANNELS, Samples = v.NUM_SAMPLES, 
                                 dropoutRate = 0.5)
            
    # compile the model and set the optimizers
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
                metrics = ['accuracy'])

    # count number of parameters in the model
    numParams    = model.count_params()    

    # set a valid path for your system to record model checkpoints
    checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                                save_best_only=True)

    class_weights = {0:1, 1:3}

    # fit the model
    history = model.fit(train_data, train_labels, batch_size = 32, epochs = 200, 
                            verbose = 2, validation_data=(val_data, val_labels),
                            callbacks=[checkpointer], class_weight = class_weights)

    # load optimal weights
    model.load_weights('/tmp/checkpoint.h5')

    # make prediction on test set.
    probs       = model.predict(test_data)
    preds       = probs.argmax(axis = -1)  
    acc         = np.mean(preds == test_labels)
    print("Classification accuracy: %f " % (acc))

    # print performance
    performance = compute_metrics(test_labels, preds)
    print("Accuracy, Sensitivity, Specificyty:\n")
    print(performance)

    
    # Plot Loss/Accuracy over time
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(go.Scatter( y=history.history['val_loss'], name="val_loss"), secondary_y=False)
    fig.add_trace(go.Scatter( y=history.history['loss'], name="loss"), secondary_y=False)
    fig.add_trace(go.Scatter( y=history.history['val_accuracy'], name="val accuracy"), secondary_y=True)
    fig.add_trace(go.Scatter( y=history.history['accuracy'], name="accuracy"), secondary_y=True)
    # Add figure title
    fig.update_layout(title_text="Loss/Accuracy of ShallowConvNet")
    # Set x-axis title
    fig.update_xaxes(title_text="Epoch")
    # Set y-axes titles
    fig.update_yaxes(title_text="Loss", secondary_y=False)
    fig.update_yaxes(title_text="Accuracy", secondary_y=True)
    fig.show()


    # plot the confusion matrices for both classifiers
    conf_matrix = metrics.confusion_matrix(test_labels,preds)
    print(conf_matrix)
    return probs

def kfold_ShallowConvNet_classification(train_data, test_data, train_labels, test_labels, n_folds, data_type, epoched = True):
    
    if epoched:
        if data_type == 'new_ica':
            model = ShallowConvNet(nb_classes = 2, Chans = v.NUM_CHANNELS, Samples = v.EPOCH_LENGTH*v.NEW_SFREQ, 
                                dropoutRate = 0.5)
        else:
            model = ShallowConvNet(nb_classes = 2, Chans = v.NUM_CHANNELS, Samples = v.EPOCH_LENGTH*v.SFREQ, 
                                dropoutRate = 0.5)
    else: #if not epoched
        if data_type == 'new_ica':
           model = ShallowConvNet(nb_classes = 2, Chans = v.NUM_CHANNELS, Samples = v.NEW_NUM_SAMPLES, 
                                dropoutRate = 0.5)
        else:
            model = ShallowConvNet(nb_classes = 2, Chans = v.NUM_CHANNELS, Samples = v.NUM_SAMPLES, 
                                 dropoutRate = 0.5)
            
    # compile the model and set the optimizers
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
                metrics = ['accuracy'])

    # count number of parameters in the model
    numParams    = model.count_params()    

    # set a valid path for your system to record model checkpoints
    checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                                save_best_only=True)

    class_weights = {0:1, 1:3}

     # Split into k-folds
    skf = StratifiedKFold(n_splits=n_folds)
    total_accuracy = 0

    for fold, (train_index, val_index) in enumerate(skf.split(train_data, train_labels)):
        print(f"\nFold nr: {fold+1}")
        train_data_fold, train_labels_fold = train_data[train_index], train_labels[train_index]
        val_data_fold, val_labels_fold = train_data[val_index], train_labels[val_index]

        history = model.fit(train_data_fold, train_labels_fold, batch_size = 16, epochs = 100, 
                            verbose = 2, validation_data=(val_data_fold, val_labels_fold),
                            callbacks=[checkpointer], class_weight = class_weights)

        # load optimal weights
        model.load_weights('/tmp/checkpoint.h5')

        probs       = model.predict(test_data)
        preds       = probs.argmax(axis = -1)  
        acc         = np.mean(preds == test_labels)
        total_accuracy += acc
        print("Classification accuracy: %f " % (acc))

        print(classification_report(test_labels, preds))
        
        # Plot Loss/Accuracy over time
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Add traces
        fig.add_trace(go.Scatter( y=history.history['val_loss'], name="val_loss"), secondary_y=False)
        fig.add_trace(go.Scatter( y=history.history['loss'], name="loss"), secondary_y=False)
        fig.add_trace(go.Scatter( y=history.history['val_accuracy'], name="val accuracy"), secondary_y=True)
        fig.add_trace(go.Scatter( y=history.history['accuracy'], name="accuracy"), secondary_y=True)
        # Add figure title
        fig.update_layout(title_text="Loss/Accuracy of k-fold ShallowConvNet")
        # Set x-axis title
        fig.update_xaxes(title_text="Epoch")
        # Set y-axes titles
        fig.update_yaxes(title_text="Loss", secondary_y=False)
        fig.update_yaxes(title_text="Accuracy", secondary_y=True)
        fig.show()
    
    classification_acc = total_accuracy/n_folds
    print(f"Shallow overall classification accuracy: {classification_acc}")
