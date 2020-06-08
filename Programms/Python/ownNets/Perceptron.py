import numpy as np

from ownNets.NN import MyNeuronalNet, flatten_input


class Perceptron(MyNeuronalNet):
    def __init__(self, name="MyPerceptron", input_size=784, output_size=10, learning_speed=0.01, train_epoch=10):
        super().__init__(name, input_size, output_size, learning_speed, train_epoch)
        self.weights = np.random.rand(output_size, input_size)

    def evaluate(self, input_data):
        return self.weights @ input_data.flatten()  # matrix multiplication

    def train_once(self, input_data, output_data):
        input_data = input_data.flatten()
        result = self.evaluate(input_data)

        # delta rule
        delta = output_data - result
        gradient = np.transpose(np.transpose(input_data[np.newaxis]) @ delta[np.newaxis])
        self.weights += self.learning_speed * gradient


