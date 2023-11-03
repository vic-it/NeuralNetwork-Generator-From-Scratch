import numpy as np

from network.layer import Layer


class ReLU(Layer):

    def forward(self, input):
        self.input = input
        # leaky RELU
        return np.maximum(0.1 * self.input, self.input)
        # return np.maximum(0, self.input)

    def backward(self, output_gradient, learning_rate):
        # print("before relu grad: ", self.input)
        a = np.greater(self.input, 0).astype(int)
        a = np.where(a <= 0, 0.1, a)
        # print("after relu grad: ", a)
        self.output_gradient = output_gradient.reshape(a.shape)
        gradient = np.multiply(self.output_gradient, a)
        # print("prev grad: ",self.output_gradient)
        # print("act grad: ",a)
        # print("out grad: ",np.multiply(self.output_gradient,a))
        return gradient
