import sys

import numpy as np

from network.layer import Layer


class Noise(Layer):
    def __init__(self, mean=0, std=0.2, lower_limit=-(sys.float_info.max - 1), upper_limit=sys.float_info.max):
        self.mean = mean
        self.std = std
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def forward(self, input):
        self.input = input
        out = self.input + np.random.normal(self.mean, self.std, self.input.shape)
        out = np.where(out < self.lower_limit, self.lower_limit, out)
        out = np.where(out > self.upper_limit, self.upper_limit, out)
        # add gaussian noise over every entry
        return out

    def backward(self, output_gradient, learning_rate):
        return output_gradient
