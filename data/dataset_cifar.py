import os

import numpy as np
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split


def get_data(test_size_percent):
    X_train, y_train, X_test, y_test = fetch()
    print("preprocessing")
    X_train = preprocess_x(X_train)
    X_test = preprocess_x(X_test)
    y_train = preprocess_y(y_train)
    y_test = preprocess_y(y_test)
    X_test, y_test = split(test_size_percent, X_test, y_test)
    X_train, y_train = split(test_size_percent, X_train, y_train)
    print("training data size: ", len(y_train))
    print("test data size: ", len(y_test))
    return X_train, y_train, X_test, y_test, X_train.shape[1] * X_train.shape[2]* X_train.shape[3], len(y_train[0])


def split(test_size_percent, X, y):
    # reduce sample size
    _, X, _, y = train_test_split(X, y, test_size=test_size_percent)
    return X, y
    # return train_test_split(X, y, test_size=test_size_percent)


def preprocess_y(y):
    z = []
    for value in y:
        l = np.zeros(10)
        l[value] = 1
        z.append(l)
    return z


def preprocess_x(X):
    X = X / 254.
    return X


def fetch():
    if os.path.exists("data/files/cifar_y_train.npy") and os.path.exists('data/files/cifar_x_train.npy'):
        print("loading from file")
        y_train = np.load('data/files/cifar_y_train.npy')
        X_train = np.load('data/files/cifar_x_train.npy')
        y_test = np.load('data/files/cifar_y_test.npy')
        X_test = np.load('data/files/cifar_x_test.npy')
        return X_train, y_train, X_test, y_test

    print("loading from web")
    # fetch dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print("saving")
    np.save('data/files/cifar_y_train', y_train)
    np.save('data/files/cifar_y_test', y_test)
    np.save('data/files/cifar_x_train', X_train)
    np.save('data/files/cifar_x_test', X_test)

    return X_train, y_train, X_test, y_test