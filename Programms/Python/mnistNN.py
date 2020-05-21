import tensorflow as tf
from tensorflow import keras
import sys
from Gui import Gui
from NN_wrapper import NN

print(sys.version)
print(tf.__version__)

gui = Gui()

# creating the nn
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # pure input transform, 2d to 1d
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(128, activation='sigmoid'),  # layer with 128 nodes
    keras.layers.Dense(10),  # output layer (1 node = 1 class)
    keras.layers.Softmax()  # visual presentation
])

nn = NN(name="firstNN", model=model, train_epoch=10)
gui.add_model(nn)

# creating the nn
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # pure input transform, 2d to 1d
    keras.layers.Dense(10),  # output layer (1 node = 1 class)
    keras.layers.Softmax()  # visual presentation
])

nn = NN(name="noHidden", model=model, train_epoch=10)
gui.add_model(nn)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # pure input transform, 2d to 1d
    keras.layers.Dense(1, activation='sigmoid'),  # layer with 128 nodes
    keras.layers.Dense(10),  # output layer (1 node = 1 class)
    keras.layers.Softmax()  # visual presentation
])
nn = NN(name="singleConnect", model=model, train_epoch=10)
gui.add_model(nn)

# creating the nn
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # pure input transform, 2d to 1d
    keras.layers.Dense(128, activation='sigmoid'),  # layer with 128 nodes
    keras.layers.Dense(10),  # output layer (1 node = 1 class)
    keras.layers.Softmax()  # visual presentation
])

nn = NN(name="firstNNnoDropout", model=model, train_epoch=10)
gui.add_model(nn)

# creating the nn
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # pure input transform, 2d to 1d
    keras.layers.Dense(10, activation='sigmoid'),  # layer with 128 nodes
    keras.layers.Dense(10),  # output layer (1 node = 1 class)
    keras.layers.Softmax()  # visual presentation
])

nn = NN(name="tenConnect", model=model, train_epoch=10)
gui.add_model(nn)

# creating the nn
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # pure input transform, 2d to 1d
    keras.layers.Dense(1024, activation='sigmoid'),  # layer with 128 nodes
    keras.layers.Dense(10),  # output layer (1 node = 1 class)
    keras.layers.Softmax()  # visual presentation
])

nn = NN(name="large2nd", model=model, train_epoch=10)
gui.add_model(nn)
# creating the nn
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # pure input transform, 2d to 1d
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(128, activation='sigmoid'),  # layer with 128 nodes
    keras.layers.Dense(10),  # output layer (1 node = 1 class)
    keras.layers.Softmax()  # visual presentation
])

nn = NN(name="firstNN100Epoch", model=model, train_epoch=100)
gui.add_model(nn)

# creating the nn
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # pure input transform, 2d to 1d
    keras.layers.Dense(128, activation='sigmoid'),  # layer with 128 nodes
    keras.layers.Dense(128, activation='sigmoid'),  # layer with 128 nodes
    keras.layers.Dense(128, activation='sigmoid'),  # layer with 128 nodes
    keras.layers.Dense(128, activation='sigmoid'),  # layer with 128 nodes
    keras.layers.Dense(128, activation='sigmoid'),  # layer with 128 nodes
    keras.layers.Dense(128, activation='sigmoid'),  # layer with 128 nodes
    keras.layers.Dense(128, activation='sigmoid'),  # layer with 128 nodes
    keras.layers.Dense(128, activation='sigmoid'),  # layer with 128 nodes
    keras.layers.Dense(128, activation='sigmoid'),  # layer with 128 nodes
    keras.layers.Dense(128, activation='sigmoid'),  # layer with 128 nodes
    keras.layers.Dense(10),  # output layer (1 node = 1 class)
    keras.layers.Softmax()  # visual presentation
])

nn = NN(name="tenLayers", model=model, train_epoch=10)
gui.add_model(nn)

# creating the nn
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # pure input transform, 2d to 1d
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(128, activation='sigmoid'),  # layer with 128 nodes
    keras.layers.Dense(10),  # output layer (1 node = 1 class)
    keras.layers.Softmax()  # visual presentation
])

nn = NN(name="oneEpoch", model=model, train_epoch=1)
gui.add_model(nn)

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
print("Accuracies:")
for i in range(len(gui.models)):
    print(gui.models[i].name + ": " + str(gui.models[i].accuracy) + " in " + str(gui.models[i].training_time))

gui.start()

exit(0)
