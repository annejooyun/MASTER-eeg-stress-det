import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from keras import models, Input
from keras import optimizers as opt
from keras import backend as K
from keras.layers import Dense
from keras_tuner.tuners import RandomSearch
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# EEGNet-specific imports
from EEGModels import EEGNet,EEGNet_SSVEP,TSGLEEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint


# PyRiemann imports
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.viz import plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression


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






def cnn_classification(train_data, test_data, train_labels, test_labels):
    model = models.Sequential()
    model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(8, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2))

    model.compile(optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy'])

    history = model.fit(train_data, train_labels, epochs=10, 
                        validation_data=(test_data, test_labels))
    
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
    print(f"Test accuracy: {test_acc}")








def EEGNet_classification(train_data, test_data, val_data, train_labels, test_labels, val_labels):

    # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other 
    # model configurations may do better, but this is a good starting point)
    model = EEGNet(nb_classes = 2, Chans = v.NUM_CHANNELS , Samples = v.NUM_SAMPLES, 
                dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
                dropoutType = 'Dropout')

    # compile the model and set the optimizers
    model.compile(loss='binary_crossentropy', optimizer='adam', 
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
    class_weights = {0:1, 1:1}

    ################################################################################
    # fit the model. Due to very small sample sizes this can get
    # pretty noisy run-to-run, but most runs should be comparable to xDAWN + 
    # Riemannian geometry classification (below)
    ################################################################################
    fittedModel = model.fit(train_data, train_labels, batch_size = 16, epochs = 300, 
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
    acc         = np.mean(preds == test_labels.argmax(axis=-1))
    print("Classification accuracy: %f " % (acc))


    ############################# PyRiemann Portion ##############################

    # code is taken from PyRiemann's ERP sample script, which is decoding in 
    # the tangent space with a logistic regression

    n_components = 2  # pick some components

    # set up sklearn pipeline
    clf = make_pipeline(XdawnCovariances(n_components),
                        TangentSpace(metric='riemann'),
                        LogisticRegression())

    preds_rg     = np.zeros(len(test_labels))

    # reshape back to (trials, channels, samples)
    train_data      = train_data.reshape(train_data.shape[0], v.NUM_CHANNELS, v.NUM_SAMPLES)
    test_data       = test_data.reshape(test_data.shape[0], v.NUM_CHANNELS, v.NUM_SAMPLES)

    # train a classifier with xDAWN spatial filtering + Riemannian Geometry (RG)
    # labels need to be back in single-column format
    clf.fit(train_data, train_labels.argmax(axis = -1))
    preds_rg     = clf.predict(test_data)

    # Printing the results
    acc2         = np.mean(preds_rg == test_labels.argmax(axis = -1))
    print("Classification accuracy: %f " % (acc2))
    """
    # plot the confusion matrices for both classifiers
    names        = ['Stressed', 'Non-stressed']
    plt.figure(0)
    plot_confusion_matrix(preds, test_labels.argmax(axis = -1), names, title = 'EEGNet-8,2')

    plt.figure(1)
    plot_confusion_matrix(preds_rg, test_labels.argmax(axis = -1), names, title = 'xDAWN + RG')
    """



def EEGNet_classification_2(train_data, test_data, val_data, train_labels, test_labels, val_labels):

    # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other 
    # model configurations may do better, but this is a good starting point)
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
    class_weights = {0:1, 1:1}

    ################################################################################
    # fit the model. Due to very small sample sizes this can get
    # pretty noisy run-to-run, but most runs should be comparable to xDAWN + 
    # Riemannian geometry classification (below)
    ################################################################################
    fittedModel = model.fit(train_data, train_labels, batch_size = 64, epochs = 300, 
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


    return probs
 
def EEGNet_classification_3(train_data, test_data, val_data, train_labels, test_labels, val_labels):

    # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other 
    # model configurations may do better, but this is a good starting point)
    model = EEGNet_SSVEP(nb_classes = 2, Chans = v.NUM_CHANNELS, Samples = v.NUM_SAMPLES, 
             dropoutRate = 0.5, kernLength = 256, F1 = 96, 
             D = 1, F2 = 96, dropoutType = 'Dropout')

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
    class_weights = {0:1, 1:1}

    ################################################################################
    # fit the model. Due to very small sample sizes this can get
    # pretty noisy run-to-run, but most runs should be comparable to xDAWN + 
    # Riemannian geometry classification (below)
    ################################################################################
    fittedModel = model.fit(train_data, train_labels, batch_size = 16, epochs = 100, 
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


    # plot the confusion matrices for both classifiers
    names        = ['Stressed', 'Non-stressed']
    plt.figure(0)
    plot_confusion_matrix(preds, test_labels, names, title = 'EEGNet-8,2')

    return probs

def EEGNet_classification_TSGL(train_data, test_data, val_data, train_labels, test_labels, val_labels):

    # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other 
    # model configurations may do better, but this is a good starting point)
    model = TSGLEEGNet(nb_classes = 2, Chans = v.NUM_CHANNELS , Samples = v.NUM_SAMPLES, 
                       dropoutRate=0.5, kernLength=64, F1=9, D=4, F2=32, FSLength=16, l1=1e-4, l21=1e-4, tl1=1e-5, norm_rate=0.25, 
                       dtype=tf.float32, dropoutType='Dropout')

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
    class_weights = {0:1, 1:1}

    ################################################################################
    # fit the model. Due to very small sample sizes this can get
    # pretty noisy run-to-run, but most runs should be comparable to xDAWN + 
    # Riemannian geometry classification (below)
    ################################################################################
    fittedModel = model.fit(train_data, train_labels, batch_size = 16, epochs = 100, 
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


    # plot the confusion matrices for both classifiers
    names        = ['Stressed', 'Non-stressed']
    plt.figure(0)
    plot_confusion_matrix(preds, test_labels, names, title = 'EEGNet-8,2')

    return probs