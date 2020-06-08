import time
from os import path

import numpy as np

import tensorflow as tf
from tensorflow import keras

save_path = 'permanency/models/'
mnist = keras.datasets.mnist

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

# preprocessing to get everything between 0 and 1
train_data = train_data / 255.0
test_data = test_data / 255.0


class NN:
    def __init__(self, name="default_name", train_epoch=10):
        self.train_epoch = train_epoch
        self.name = name

    def predict(self, array):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError


class TensorFlowNN(NN):
    def __init__(self, model, name="default_name", train_epoch=10, optimizer='adam',
                 loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy']):
        super().__init__(name, train_epoch)
        self.metrics = metrics
        self.loss = loss
        self.optimizer = optimizer
        self.model = model
        self.training_time = time.time()
        self.accuracy = 0

    def predict(self, array):
        fake_batch = np.array([array])
        prediction = self.model.predict(fake_batch)
        return np.argmax(prediction)

    def train(self):
        if path.isdir(str(save_path + self.name)):
            print("Loaded NN " + self.name + " from memory, not retrained")
            self.model = tf.keras.models.load_model(save_path + self.name)
            self.training_time = None
        else:
            self.training_time = time.time()
            self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
            self.model.fit(train_data, train_labels, epochs=self.train_epoch)  # epochs = number of iterations
            self.training_time = time.time() - self.training_time
            self.model.save(str(save_path + self.name))
        loss, self.accuracy = self.model.evaluate(test_data, test_labels, verbose=2)
