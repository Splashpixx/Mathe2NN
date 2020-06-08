import numpy as np


class MyNeuronalNet:
    def evaluate(self, input_data):
        raise NotImplementedError

    def train(self, input_data, output_data):
        raise NotImplementedError

    def batch_train(self, input_data, output_data):
        print("started training")
        for i in range(len(input_data)):
            self.train(input_data[i], output_data[i])

    def classify(self, input_data):
        result = self.evaluate(input_data)
        return np.argmax(result)

    def test(self, input_data, output_data):
        num_of_correct = 0
        for i in range(len(input_data)):
            if self.classify(input_data[i]) == output_data[i]:
                num_of_correct += 1
        return num_of_correct/input_data.length
