import numpy as np

from network.layer import Layer


class Dense(Layer):
    def __init__(self, input_size, output_size, momentum=0, layer_learning_rate = 1):
        self.last_gradient=np.zeros((output_size, input_size))
        self.last_bias = 0
        self.momentum = momentum
        self.layer_learning_rate = layer_learning_rate
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.bias = np.random.randn(output_size, 1) * 0.01

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        # previous gradient times the input values is the gradient w.r.t the weights as a vector (weight gradients)
        weights_gradient = np.dot(output_gradient, self.input.T)
        output = np.dot(self.weights.T, output_gradient)
        # adjust all weights for this layer with weights_gradients and gradient descent
        b = self.weights.copy()
        self.last_gradient = learning_rate * self.layer_learning_rate * weights_gradient + self.momentum * self.last_gradient
        self.weights -= self.last_gradient
        # adjust all biases (since bias derivative is 1 it only needs to be adjusted by the gradient from the following layer)
        self.last_bias = learning_rate * output_gradient * self.layer_learning_rate + self.momentum * self.last_bias
        self.bias -= learning_rate * output_gradient * self.layer_learning_rate
        # return derivative with respect to inputs X, so previous layer can do the same
        return output
