import numpy as np


def mse(y_true, y_pred):
    # print('true: ', y_true)
    # print('pred: ', y_pred)
    # print('mse: ', np.mean(np.power(y_pred - y_true, 2)))
    return np.mean(np.power(y_true - y_pred, 2))


def mse_gradient(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


def is_correctly_classified(y_true, y_pred):
    # print('true: ', y_true)
    # print('pred: ', y_pred)
    # print(np.argmax(y_true) == np.argmax(y_pred).astype(int))
    return np.argmax(y_true) == np.argmax(y_pred).astype(int)


def softmax(input):
    c = np.max(input)
    return np.exp(input - c) / np.sum(np.exp(input - c))


def cross_entropy(y_true, y_pred):  # CE
    return -np.sum(y_true * np.log(y_pred + 10 ** -30))


def cross_entropy_gradient(y_true, y_pred):  # CE derivative
    return y_pred - y_true
