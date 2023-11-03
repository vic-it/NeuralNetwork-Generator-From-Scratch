from data import data_generator
from network.relu import ReLU
from network.dense import Dense
from network.network import Network
from network.noise import Noise
import pickle
from network.softmax import Softmax


def start():
    X_train, y_train, X_test, y_test, in_dim, out_dim = data_generator.get_data("fashion", .99)
    layers = [Noise(0,0.2),
              Dense(in_dim, 99,0.8),
              ReLU(),
              Dense(99, 45,0.8),
              ReLU(),
              Dense(45, 15,0.8),
              ReLU(),
              Dense(15, out_dim,0.8)
              ]

    layers = load_layers("fashion.layer")
    layers[0].std = 0.2
    #softmax + cross entropy for classification
    #mse for regression
    network = Network(layers, "cross_entropy",in_dim, out_dim)
    network.train(X_train, y_train, X_test, y_test, 25, 0.0003, in_dim, 1)
    save_layers(layers,"fashion.layer")


def save_layers(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_layers(path):
    try:
        with open(path, "rb") as f:
            layer = pickle.load(f)
    except ValueError as e:
        print(e)
        layer = []
    return layer
start()
