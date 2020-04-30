import tensorflow as tf
from tensorflow import keras
from os import path
import sys

import tkinter as tk
from PIL import Image
import numpy as np
from scipy import ndimage
import io
import math

print(sys.version)
print(tf.__version__)

save_path = 'permanency/models'
model = None
if path.isdir(save_path):
    # loading model
    model = tf.keras.models.load_model(save_path)
else:
    # create new model

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

    # training the model
    model.fit(train_data, train_labels, epochs=20)  # epochs = number of iterations

    # verifying model
    test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)

    # save model
    model.save(save_path)

    print('\nVerification accuracy:', test_acc)

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

lastX, lastY = 0, 0
line_width = 10


def xy(event):
    """Takes the coordinates of the mouse when you click the mouse"""
    global lastX, lastY
    lastX, lastY = event.x, event.y


def add_line(event):
    """Creates a line when you drag the mouse
    from the point where you clicked the mouse to where the mouse is now"""
    global lastX, lastY
    canvas.create_line(lastX, lastY, event.x, event.y, width=line_width, tags='drawing')
    # this makes the new starting point of the drawing
    lastX, lastY = event.x, event.y


def evaluate(event):
    """Uses the NN to guess the currently drawn number and updates the result_label accordingly"""
    postscript = canvas.postscript(colormode='gray')
    im = Image.open(io.BytesIO(postscript.encode('utf-8')))
    im = im.convert(mode='L')
    array = np.asarray(im)
    # find section with number
    array = array - 255  # 255 == code for white
    rows, cols = np.nonzero(array)
    min_x = cols.min()
    max_x = cols.max()
    min_y = rows.min()
    max_y = rows.max()
    x_size = max_x - min_x + 1  # both ends inclusive => +1
    y_size = max_y - min_y + 1
    # crop number down to 20x<20 or <20x20, whichever is possible while conserving aspect ratio
    im = im.crop((min_x, min_y, max_x, max_y))
    if x_size > y_size:
        im = im.resize((20, math.ceil(20 * y_size / x_size)))
    else:
        im = im.resize((math.ceil(20 * x_size / y_size), 20))
    # find center of mass (COM) of the number
    array = np.asarray(im)
    array = 255 - array  # invert image, library searches for COM of white
    center_y, center_x = ndimage.measurements.center_of_mass(array)  # yes, library returns a (y, x) tuple
    center_x = int(center_x)
    center_y = int(center_y)
    horizontal_offset = 14 - center_x
    vertical_offset = 14 - center_y
    # shift image so that COM is in the center of a 28x28 frame
    blank = Image.new("L", (28, 28), color=255)
    blank.paste(im, (horizontal_offset, vertical_offset))
    im = blank
    # feed image into NN
    array = np.asarray(im)
    array = 255 - array  # invert image - Pillow and MNIST don't agree whether 0 is white or black
    fake_batch = np.array([array])
    prediction = model.predict(fake_batch)
    # update label to show new recognised number
    result = np.argmax(prediction)
    certainty = 100 * np.max(prediction)
    certainty_string = format(certainty, '.2f')
    canvas.itemconfig(result_label, text=('' + str(result) + ' (' + str(certainty_string) + '%)'))
    print("result: ", result)
    print("certainty: ", certainty, "%")


def line_width_increase(event):
    """increases line width by 5"""
    global line_width
    line_width = line_width + 5


def line_width_decrease(event):
    """decreases line width by 5, but never below 5"""
    global line_width
    if line_width > 5:
        line_width -= 5


def delete_current_drawing(event):
    """deletes everything tagged 'drawing'"""
    canvas.delete('drawing')
    print("delete_current_drawing")


def end_program(event):
    """ends the program"""
    exit(0)


root = tk.Tk()
root.geometry("800x800")
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

canvas = tk.Canvas(root)
canvas.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
canvas.bind("<Button-1>", xy)
canvas.bind("<B1-Motion>", add_line)
canvas.bind("<ButtonRelease-1>", evaluate)
result_label = canvas.create_text(200, 100, text="Nothing drawn yet")
root.bind("+", line_width_increase)
root.bind("-", line_width_decrease)
root.bind("<Escape>", end_program)
root.bind("<BackSpace>", delete_current_drawing)

root.mainloop()
exit(0)
