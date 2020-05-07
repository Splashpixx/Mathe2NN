import time
from sys import path

import numpy as np

from mnistNN import save_path, test_data, test_labels, train_data, train_labels
import tensorflow as tf


class NN:
    def __init__(self, model, name="default_name", train_epoch=20):
        self.name = name
        self.train_epoch = train_epoch
        self.training_time = time.time()
        if path.isdir(save_path + name):
            print("Loaded NN " + name + " from memory, not retrained")
            self.model = tf.keras.models.load_model(save_path + name)
            self.training_time = None
        else:
            self.model = model
            self.train()
            self.training_time = time.time() - self.training_time
        loss, self.accuracy = model.evaluate(test_data, test_labels, verbose=2)
        model.save(save_path + self.name)

    def predict(self, array):
        fake_batch = np.array([array])
        prediction = self.model.predict(fake_batch)
        return np.argmax(prediction)

    def train(self):
        self.model.fit(train_data, train_labels, epochs=self.train_epoch)  # epochs = number of iterations
