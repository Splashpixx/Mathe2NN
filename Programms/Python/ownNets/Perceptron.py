import numpy as np

from ownNets.NN import MyNeuronalNet


class Perceptron(MyNeuronalNet):
    def __init__(self, input_size=784, output_size=10, learning_speed=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_speed = learning_speed
        self.weights = np.random.rand(output_size, input_size)

    def evaluate(self, input_data):
        return self.weights @ input_data  # matrix multiplication

    def train(self, input_data, output_data):
        result = self.evaluate(input_data)

        # delta rule
        delta = output_data - result
        gradient = delta * input_data  # element wise multiplication
        self.weights += self.learning_speed * gradient


