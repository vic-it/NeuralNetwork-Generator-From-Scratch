import os

import numpy as np
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


def get_data(test_size_percent):
    X, y = fetch()
    X_train, X_test, y_train, y_test = split(test_size_percent, X, y)
    print("preprocessing")
    y_train = preprocess_y(y_train)
    y_test = preprocess_y(y_test)
    print("training data size: ", len(y_train))
    print("test data size: ", len(y_test))
    return X_train, y_train, X_test, y_test, len(X_train[0]), len(y_train[0])


def split(test_size_percent, X, y):
    # reduce sample size
    # _, X, _, y = train_test_split(X, y, test_size=1)
    return train_test_split(X, y, test_size=test_size_percent)


def preprocess_y(y):
    dict = {'Iris-setosa': [1., 0., 0.], 'Iris-versicolor': [0., 1., 0.], 'Iris-virginica': [0., 0., 1.]}
    y = [dict[a[0]] for a in y]
    return y


def fetch():
    if os.path.exists("data/files/iris_y.npy") and os.path.exists("data/files/iris_x.npy"):
        print("loading from file")
        y = np.load('data/files/iris_y.npy', allow_pickle=True)
        X = np.load('data/files/iris_x.npy', allow_pickle=True)
        return X, y

    print("loading from web")
    # fetch dataset
    iris = fetch_ucirepo(id=53)
    # data (as pandas dataframes)
    X = iris.data.features.to_numpy()
    y = iris.data.targets.to_numpy()
    print("saving")
    np.save('data/files/iris_y', y)
    np.save('data/files/iris_x', X)

    return X, y
