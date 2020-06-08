import numpy as np

from ownNets.NN import MyNeuronalNet

import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


class FeedForward(MyNeuronalNet):
    def __init__(self, input_size=784, output_size=10, num_of_hidden_layers=1,
                 hidden_layer_size=100, activation=sigmoid, activation_derivative=d_sigmoid, learning_speed=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_activation = np.zeros((num_of_hidden_layers, hidden_layer_size))
        self.hidden_layer_weights = np.random.rand(num_of_hidden_layers - 1, hidden_layer_size,
                                                   hidden_layer_size) * 2 - 1
        self.into_hidden_weights = np.random.rand(hidden_layer_size, input_size) * 2 - 1
        self.out_of_hidden_weights = np.random.rand(output_size, hidden_layer_size) * 2 - 1
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.learning_speed = learning_speed
        self.hidden_layer_size = hidden_layer_size
        self.num_of_hidden_layers = num_of_hidden_layers

    def evaluate(self, input_data):
        self.hidden_layer_activation[0] = self.activation(self.into_hidden_weights @ input_data)
        for i in range(1, self.num_of_hidden_layers - 1):
            self.hidden_layer_activation[i] = \
                self.activation(self.hidden_layer_weights[i] @ self.hidden_layer_activation[i - 1])
        output = self.activation(self.out_of_hidden_weights @ self.hidden_layer_activation[1])
        self.hidden_layer_activation = self.activation_derivative(self.hidden_layer_activation)
        return output

    def train(self, input_data, output_data):
        result = self.evaluate(input_data)
        # Backpropagation
        delta = np.zeros((self.num_of_hidden_layers, self.hidden_layer_size))

        # find delta for output neurons
        delta_output = self.activation_derivative(output_data - result)

        # find d for the last hidden layer
        for hidden_neuron in range(self.hidden_layer_size):
            delta[self.num_of_hidden_layers-2, hidden_neuron] = self.out_of_hidden_weights @ delta_output # to do
        pass
