import os
import random

import numpy as np
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


def get_data(test_size_percent):
    X, y = fetch()
    X_train, X_test, y_train, y_test = split(test_size_percent, X, y)
    X_train, y_train = unskew(X_train, y_train)
    y_train = preprocess_y(y_train)
    y_test = preprocess_y(y_test)
    print("training data size: ", len(y_train))
    print("test data size: ", len(y_test))
    return X_train, y_train, X_test, y_test, len(X_train[0]), len(y_train[0])


def split(test_size_percent, X, y):
    # reduce sample size
    _, X, _, y = train_test_split(X, y, test_size=0.050)
    return train_test_split(X, y, test_size=test_size_percent)


def unskew(X_train, y_train):
    nonzero_train = np.count_nonzero(y_train)
    print("BEFORE UNSKEWING: ")
    print("diabetics: ", np.count_nonzero(y_train))
    print("non diabetics: ", len(y_train) - np.count_nonzero(y_train))
    remove_ratio = (len(y_train) - 2 * nonzero_train) / (len(y_train) - nonzero_train)
    for i in reversed(range(len(y_train))):
        e = y_train[i][0]
        should_delete = random.uniform(0, 1) < remove_ratio
        if e == 0 and should_delete:
            y_train = np.delete(y_train, i, 0)
            X_train = np.delete(X_train, i, 0)
    print("AFTER UNSKEWING: ")
    print("diabetics: ", np.count_nonzero(y_train))
    print("non diabetics: ", len(y_train) - np.count_nonzero(y_train))
    return X_train, y_train


def preprocess_y(y):
    dict = {0: [1., 0.], 1: [0., 1.]}
    y = [dict[a[0]] for a in y]
    return y


def fetch():
    if os.path.exists("data/files/diabetes_y.npy") and os.path.exists("data/files/diabetes_x.npy"):
        print("loading from file")
        y = np.load('data/files/diabetes_y.npy')
        X = np.load('data/files/diabetes_x.npy')
        return X, y

    print("loading from web")
    # fetch dataset
    cdc_diabetes_health_indicators = fetch_ucirepo(id=891)
    # data (as pandas dataframes)
    X = cdc_diabetes_health_indicators.data.features.to_numpy()
    y = cdc_diabetes_health_indicators.data.targets.to_numpy()
    print("saving")
    np.save('data/files/diabetes_y', y)
    np.save('data/files/diabetes_x', X)
    return X, y
