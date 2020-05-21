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
    def __init__(self, model, name="default_name", train_epoch=20, optimizer='adam',
                 loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy']):
        self.metrics = metrics
        self.loss = loss
        self.optimizer = optimizer
        self.name = name
        self.train_epoch = train_epoch
        self.training_time = time.time()
        if path.isdir(str(save_path + name)):
            print("Loaded NN " + name + " from memory, not retrained")
            self.model = tf.keras.models.load_model(save_path + name)
            self.training_time = None
        else:
            self.model = model
            self.train()
            self.training_time = time.time() - self.training_time
            model.save(str(save_path + name))
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        loss, self.accuracy = model.evaluate(test_data, test_labels, verbose=2)

    def predict(self, array):
        fake_batch = np.array([array])
        prediction = self.model.predict(fake_batch)
        return np.argmax(prediction)

    def train(self):
        self.model.fit(train_data, train_labels, epochs=self.train_epoch)  # epochs = number of iterations
