import numpy as np

from ownNets.NN import MyNeuronalNet

import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


class FeedForward(MyNeuronalNet):
    def __init__(self, name="MyFeedForward", input_size=784, output_size=10, num_of_hidden_layers=1,
                 hidden_layer_size=100,
                 activation=sigmoid, activation_derivative=d_sigmoid, learning_speed=0.001):
        super().__init__(name, input_size, output_size, learning_speed)
        self.hidden_layer_activation = np.zeros((num_of_hidden_layers, hidden_layer_size))
        self.hidden_layer_weights = np.random.rand(num_of_hidden_layers - 1, hidden_layer_size,
                                                   hidden_layer_size) * 2 - 1
        self.into_hidden_weights = np.random.rand(hidden_layer_size, input_size) * 2 - 1
        self.out_of_hidden_weights = np.random.rand(output_size, hidden_layer_size) * 2 - 1
        self.activation = np.vectorize(activation)
        self.activation_derivative = np.vectorize(activation_derivative)
        self.hidden_layer_size = hidden_layer_size
        self.num_of_hidden_layers = num_of_hidden_layers

    def evaluate(self, input_data):
        self.hidden_layer_activation[0] = self.activation(self.into_hidden_weights @ input_data)
        for i in range(1, self.num_of_hidden_layers - 1):
            self.hidden_layer_activation[i] = \
                self.activation(self.hidden_layer_weights[i] @ self.hidden_layer_activation[i - 1])
        output = self.activation(
            self.out_of_hidden_weights @ self.hidden_layer_activation[len(self.hidden_layer_activation) - 1])
        self.hidden_layer_activation = self.activation_derivative(self.hidden_layer_activation)
        return output

    def train_once(self, input_data, output_data):
        result = self.evaluate(input_data)
        # Backpropagation
        delta = np.zeros((self.num_of_hidden_layers, self.hidden_layer_size))

        # find delta for output neurons
        delta_output = self.activation_derivative(output_data - result)

        out_of_hidden_gradient = delta_output * self.hidden_layer_activation[len(self.hidden_layer_activation) - 1]
        self.out_of_hidden_weights += self.learning_speed * out_of_hidden_gradient

        # find d for the last hidden layer
        for hidden_neuron in range(self.hidden_layer_size):
            delta[self.num_of_hidden_layers - 2, hidden_neuron] = \
                self.out_of_hidden_weights[:, hidden_neuron] @ delta_output

        # find d for the other layers
        if self.num_of_hidden_layers > 1:
            for layer in reversed(range(0, self.num_of_hidden_layers-2)):
                for neuron in range(self.hidden_layer_size):
                    delta[layer, neuron] = self.hidden_layer_weights[layer + 1] @ delta[layer + 1]

            for layer in reversed(range(self.num_of_hidden_layers-1, 1)):
                layer_gradient = delta[layer] * self.hidden_layer_activation[layer - 1]
                self.hidden_layer_weights[layer] += self.learning_speed * layer_gradient

        into_hidden_gradient = delta[0] * input
        self.into_hidden_weights += self.learning_speed * into_hidden_gradient
