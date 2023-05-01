import utils.metrics as m
from utils.EEGModels import EEGNet,TSGLEEGNet, DeepConvNet, ShallowConvNet, TSGLEEGNet
import utils.variables as v
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

import plotly.graph_objects as go
import numpy as np
import tensorflow as tf


def knn_classification(train_data, test_data, train_labels, test_labels):
    param_grid = {
        'leaf_size': range(1, 10),
        'n_neighbors': range(1, 5),
        'p': [1, 2]
    }
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    knn_clf = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, n_jobs=-1, cv = 10)
    knn_clf.fit(train_data, train_labels)

    y_pred = knn_clf.predict(test_data)
    y_true = test_labels

    print(knn_clf.best_estimator_)
    results = knn_clf.cv_results_

    # extract the relevant scores
    leaf_sizes = results['param_leaf_size'].data
    n_neighbors = results['param_n_neighbors'].data
    accuracies = results['mean_test_score']

    print('Number of results:', len(accuracies))
    #print('n_neighbors:', n_neighbors)
    #print('leaf_sizes:', leaf_sizes)
    print('accuracies:', accuracies)
    # plot the results
    plt.figure(1)
    plt.plot(
        range(len(accuracies)),
        accuracies,
    )
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.show()

    conf_matrix = metrics.confusion_matrix(y_true, y_pred)
    m.plot_conf_matrix_and_stats(conf_matrix)
    



def svm_classification(train_data, test_data, train_labels, test_labels):
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000],
        'kernel': ['rbf']
    }
    scaler = RobustScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    svm_clf = GridSearchCV(SVC(), param_grid, refit=True, n_jobs=-1, cv = 10)
    svm_clf.fit(train_data, train_labels)

    y_pred = svm_clf.predict(test_data)
    y_true = test_labels

    print(svm_clf.best_estimator_)
    # fit the grid search to get the results
    results = svm_clf.cv_results_

    # extract the relevant scores
    C_values = results['param_C'].data
    kernel_values = results['param_kernel'].data
    accuracies = results['mean_test_score']

    print('Number of results:', len(accuracies))
    #print('C_values:', C_values)
    #print('kernel_values:', kernel_values)
    print('accuracies:', accuracies)
    # plot the results
    plt.figure(2)
    plt.plot(
        range(len(accuracies)),
        accuracies
    )
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.show()
    
    conf_matrix = metrics.confusion_matrix(y_true, y_pred)
    m.plot_conf_matrix_and_stats(conf_matrix)



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
    history = model.fit(train_data, train_labels, batch_size = None, epochs = 30, 
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

        history = model.fit(train_data_fold, train_labels_fold, batch_size = None, epochs = 30, 
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
    history = model.fit(train_data, train_labels, batch_size = None, epochs = 30, 
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

        history = model.fit(train_data_fold, train_labels_fold, batch_size = None, epochs = 30, 
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
    history = model.fit(train_data, train_labels, batch_size = None, epochs = 30, 
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

        history = model.fit(train_data_fold, train_labels_fold, batch_size = None, epochs = 30, 
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
    history = model.fit(train_data, train_labels, batch_size = None, epochs = 30, 
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

        history = model.fit(train_data_fold, train_labels_fold, batch_size = None, epochs = 30, 
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
