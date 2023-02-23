import numpy as np
from sklearn.model_selection import train_test_split, train_test_split, GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import keras
from keras.models import load_model
from keras import backend as K
from keras import models
from keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
from keras_tuner.tuners import RandomSearch
import keras_tuner

from metrics import compute_metrics
import utils.variables as v


def KNN(data, label):
    K.clear_session()
    print(data.shape)
    x, x_test, y, y_test = train_test_split(data, label, test_size=0.2, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.25, random_state=1)
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x = scaler.transform(x)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    param_grid = {
        'leaf_size': range(50),
        'n_neighbors': range(1, 10),
        'p': [1, 2]
    }
    split_index = [-1 if x in range(len(x_train)) else 0 for x in range(len(x))]
    ps = PredefinedSplit(test_fold=split_index)
    knn_clf = GridSearchCV(KNeighborsClassifier(), param_grid, cv=ps, refit=True)
    knn_clf.fit(x, y)

    y_pred = knn_clf.predict(x_test)
    y_true = y_test

    return compute_metrics(y_true, y_pred)










def SVM(data, label):
   x, x_test, y, y_test = train_test_split(data, label, test_size=0.2, random_state=1)
   x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.25, random_state=1)
   param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'kernel': ['rbf']
   }
   split_index = [-1 if x in range(len(x_train)) else 0 for x in range(len(x))]
   ps = PredefinedSplit(test_fold=split_index)
   svm_clf = GridSearchCV(SVC(), param_grid, cv=ps, refit=True)
   svm_clf.fit(x, y)
   y_pred = svm_clf.predict(x_test)
   y_true = y_test
   
   return compute_metrics(y_true, y_pred)







def NN(data, label):
    K.clear_session()
    y_v = label
    y_v = to_categorical(y_v, v.NUM_CLASSES)
    x_train, x_test, y_train, y_test = train_test_split(data, y_v, test_size=0.2, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)

    def model_builder(hp):
        model = keras.models.Sequential()
        model.add(keras.Input(shape=(x_train.shape[1],)))

        for i in range(hp.Int('layers', 2, 6)):
            model.add(keras.layers.Dense(units=hp.Int('units_' + str(i), 32, 1024, step=32),
                                        activation=hp.Choice('act_' + str(i), ['relu', 'sigmoid'])))

            model.add(keras.layers.Dense(v.NUM_CLASSES, activation='softmax', name='out'))
        
            hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

            model.compile(optimizer=keras.optimizers.adam_v2.Adam(learning_rate=hp_learning_rate),
                    loss = "binary_crossentropy",
                    metrics=['accuracy'])
            return model


    tuner = RandomSearch(
        model_builder,
        objective = 'val_accuracy',
        max_trials = 5,
        executions_per_trial = 2,
        overwrite=True
    )

    tuner.search_space_summary()

    tuner.search(x_train, y_train, epochs = 50, validation_data= [x_val, y_val])

    model = tuner.get_best_models(num_models=1)[0]

    y_pred = model.predict(x_test)
    y_true = y_test
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)

    scores_dnn = model.evaluate(x_test, y_test, verbose=0)

    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))