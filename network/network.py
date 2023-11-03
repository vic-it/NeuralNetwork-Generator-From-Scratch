from matplotlib import pyplot as plt

from network.metric import is_correctly_classified, mse, mse_gradient, cross_entropy, \
    cross_entropy_gradient
from network.relu import ReLU
from network.softmax import Softmax


class Network:

    def __init__(self, layers, loss_key, in_dim, out_dim):
        if loss_key == "mse":
            self.last_activation =ReLU()
            self.loss = mse
            self.loss_gradient = mse_gradient
        elif loss_key == "cross_entropy":
            self.last_activation = Softmax()
            self.loss = cross_entropy
            self.loss_gradient = cross_entropy_gradient
        self.loss_key = loss_key
        self.layers = layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.output = None

    def train(self, X_train, Y_train, X_test, Y_test, epochs, learning_rate, in_dim, print_rate):
        plot_data = []
        plot_data.append([])
        plot_data.append([])
        plot_data.append([])
        plot_data.append([])
        plot_data.append([])
        ctr = 0
        # train
        for e in range(epochs):
            ctr += 1
            train_accuracy = 0
            test_accuracy = 0
            train_error = 0
            test_error = 0
            for x, y in zip(X_train, Y_train):
                output = x.reshape(in_dim, 1)
                # forward
                for layer in self.layers:
                    output = layer.forward(output)
                output = self.last_activation.forward(output)
                train_error += self.loss(y, output.T[0])
                train_accuracy += is_correctly_classified(y, output.T[0])
                gradient = self.loss_gradient(y, output.T[0])
                # backward
                gradient = self.last_activation.backward(gradient, learning_rate)
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, learning_rate)
            train_error /= len(X_train)
            train_accuracy /= len(X_train)
            # test
            for x, y in zip(X_test, Y_test):
                output = x.reshape(in_dim, 1)
                for layer in self.layers:
                    output = layer.forward(output)
                output = self.last_activation.forward(output)
                test_error += self.loss(y, output.T[0])
                test_accuracy += is_correctly_classified(y, output.T[0])
            test_error /= len(X_test)
            test_accuracy /= len(X_test)
            plot_data[0].append(ctr)
            plot_data[1].append(train_error)
            plot_data[2].append(test_error)
            plot_data[3].append(train_accuracy)
            plot_data[4].append(test_accuracy)
            if (((e + 1) % print_rate) == 0) or (e == 0):
                print("%d/%d, train_error=%f - test_error=%f ||||| train_accuracy=%.1f%% - test_accuracy=%.1f%%" %
                      (e + 1, epochs, train_error, test_error, train_accuracy * 100., test_accuracy * 100.))
        self.draw_plot(plot_data)

    def classify(self, x, *y):
        output = x.reshape(self.in_dim, 1)
        for layer in self.layers:
            output = layer.forward(output)
        output = self.last_activation.forward(output)
        print("predicted: ", output)
        if len(y) > 0:
            error = self.loss(y, output.T[0])
            was_correct = is_correctly_classified(y, output.T[0])
            print("actual: ", y)
            print("correctly classified: ", was_correct)
            print("error: ", error)

    def draw_plot(self, plot_data):
        plt.plot(plot_data[0], plot_data[1], label="train error")
        plt.plot(plot_data[0], plot_data[2], label="test error")
        plt.plot(plot_data[0], plot_data[3], label="train accuracy")
        plt.plot(plot_data[0], plot_data[4], label="test accuracy")
        plt.ylabel('error/accuracy')
        plt.xlabel('epoch')
        plt.ylim(0, 1)
        plt.legend()
        plt.show()
