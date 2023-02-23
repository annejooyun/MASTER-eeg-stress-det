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