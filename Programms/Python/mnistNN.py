import tensorflow as tf
from tensorflow import keras
import sys
from NN_wrapper import NN
from Gui import Gui

print(sys.version)
print(tf.__version__)


save_path = 'permanency/models/'
gui = Gui()
mnist = keras.datasets.mnist

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

# preprocessing to get everything between 0 and 1
train_data = train_data / 255.0
test_data = test_data / 255.0

# creating the nn
model = keras.Sequential([
    keras.layers.Flatten(input_shape=train_data[0].shape),  # pure input transform, 2d to 1d
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(128, activation='sigmoid'),  # layer with 128 nodes
    keras.layers.Dense(10),  # output layer (1 node = 1 class)
    keras.layers.Softmax()  # visual presentation
])

# compiling the nn
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

nn = NN(name="firstNN", model=model, train_epoch=10)
gui.add_model(nn)

# creating the nn
model = keras.Sequential([
    keras.layers.Flatten(input_shape=train_data[0].shape),  # pure input transform, 2d to 1d
    keras.layers.Dense(10),  # output layer (1 node = 1 class)
    keras.layers.Softmax()  # visual presentation
])

# compiling the nn
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

nn = NN(name="noHidden", model=model, train_epoch=10)
gui.add_model(nn)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=train_data[0].shape),  # pure input transform, 2d to 1d
    keras.layers.Dense(1, activation='sigmoid'),  # layer with 128 nodes
    keras.layers.Dense(10),  # output layer (1 node = 1 class)
    keras.layers.Softmax()  # visual presentation
])
# compiling the nn
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
nn = NN(name="singleConnect", model=model, train_epoch=10)
gui.add_model(nn)

# creating the nn
model = keras.Sequential([
    keras.layers.Flatten(input_shape=train_data[0].shape),  # pure input transform, 2d to 1d
    keras.layers.Dense(128, activation='sigmoid'),  # layer with 128 nodes
    keras.layers.Dense(10),  # output layer (1 node = 1 class)
    keras.layers.Softmax()  # visual presentation
])

# compiling the nn
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

nn = NN(name="firstNNnoDropout", model=model, train_epoch=10)
gui.add_model(nn)

# creating the nn
model = keras.Sequential([
    keras.layers.Flatten(input_shape=train_data[0].shape),  # pure input transform, 2d to 1d
    keras.layers.Dense(10, activation='sigmoid'),  # layer with 128 nodes
    keras.layers.Dense(10),  # output layer (1 node = 1 class)
    keras.layers.Softmax()  # visual presentation
])

# compiling the nn
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

nn = NN(name="tenConnect", model=model, train_epoch=10)
gui.add_model(nn)

# creating the nn
model = keras.Sequential([
    keras.layers.Flatten(input_shape=train_data[0].shape),  # pure input transform, 2d to 1d
    keras.layers.Dense(1024, activation='sigmoid'),  # layer with 128 nodes
    keras.layers.Dense(10),  # output layer (1 node = 1 class)
    keras.layers.Softmax()  # visual presentation
])

# compiling the nn
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

nn = NN(name="large2nd", model=model, train_epoch=10)
gui.add_model(nn)
# creating the nn
model = keras.Sequential([
    keras.layers.Flatten(input_shape=train_data[0].shape),  # pure input transform, 2d to 1d
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(128, activation='sigmoid'),  # layer with 128 nodes
    keras.layers.Dense(10),  # output layer (1 node = 1 class)
    keras.layers.Softmax()  # visual presentation
])

# compiling the nn
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

nn = NN(name="firstNN100Epoch", model=model, train_epoch=100)
gui.add_model(nn)

# creating the nn
model = keras.Sequential([
    keras.layers.Flatten(input_shape=train_data[0].shape),  # pure input transform, 2d to 1d
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

# compiling the nn
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

nn = NN(name="tenLayers", model=model, train_epoch=10)
gui.add_model(nn)

# creating the nn
model = keras.Sequential([
    keras.layers.Flatten(input_shape=train_data[0].shape),  # pure input transform, 2d to 1d
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(128, activation='sigmoid'),  # layer with 128 nodes
    keras.layers.Dense(10),  # output layer (1 node = 1 class)
    keras.layers.Softmax()  # visual presentation
])

# compiling the nn
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

nn = NN(name="oneEpoch", model=model, train_epoch=1)
gui.add_model(nn)

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
print("Accuracies:")
for i in range(len(gui.models)):
    print(gui.models[i].name+": " + str(gui.models[i].accuracy) + " in " + str(gui.models[i].training_time))

gui.start()

exit(0)
