import sys

import numpy as np

from network.layer import Layer
from network.metric import softmax


class Softmax(Layer):
    def __init__(self, mean=0, std=0.2, lower_limit=-(sys.float_info.max - 1), upper_limit=sys.float_info.max):
        return

    def forward(self, input):
        self.output =softmax(input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        #IF LOSS IS CROSS ENTROPY JUST RETURN PREVIOUS GRADIENT
        return np.reshape(output_gradient, (len(output_gradient),1))
        #BELOW IS FOR LOSS GRADIENT OTHER THAN CROSS ENTROPY
        #size = np.size(self.output)
        #n = np.tile(self.output, size)
        #out = np.dot(n*(np.identity(size)-n.T),output_gradient)
        #return np.reshape(out, (len(out),1))