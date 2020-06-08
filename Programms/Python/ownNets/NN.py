import time

import numpy as np

from NN_wrapper import NN, train_data, train_labels, test_data, test_labels


def flatten_input(array):
    d_1_result = np.zeros((array.shape[0], int(array.size / array.shape[0])))
    for i in range(array.shape[0]):
        d_1_result[i] = array[i].flatten()
    return d_1_result


class MyNeuronalNet(NN):
    def __init__(self, name="InvalidMyNet", input_size=784, output_size=10, learning_speed=0.001, train_epoch=10):
        super().__init__(name, train_epoch)
        self.output_size = output_size
        self.input_size = input_size
        self.learning_speed = learning_speed
        self.training_time = time.time()
        self.accuracy = 0

    def find_accuracy(self):
        return self.test(flatten_input(test_data), test_labels)

    def evaluate(self, input_data):
        raise NotImplementedError

    def train(self):
        self.training_time = time.time()
        output_array = np.zeros((len(train_labels), self.output_size))
        d_1_input = flatten_input(train_data)
        for i in range(len(train_labels)):
            output_array[i, train_labels[i]] = 1
        for i in range(self.train_epoch):
            print("training " + self.name + " in epoch " + str(i))
            self.batch_train(d_1_input, output_array)
        self.training_time = time.time() - self.training_time
        self.accuracy = self.find_accuracy()

    def train_once(self, input_data, output_data):
        raise NotImplementedError

    def batch_train(self, input_data, output_data):
        for i in range(len(input_data)):
            self.train_once(input_data[i], output_data[i])

    def predict(self, input_data):
        result = self.evaluate(input_data)
        return np.argmax(result)

    def test(self, input_data, output_data):
        num_of_correct = 0
        for i in range(len(input_data)):
            if self.predict(input_data[i]) == output_data[i]:
                num_of_correct += 1
        return num_of_correct / len(input_data)
