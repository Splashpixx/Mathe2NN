from tkinter import NE, NW

import tensorflow as tf
from tensorflow import keras
from os import path
import sys

import tkinter as tk
from PIL import Image, ImageTk
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


class Gui:
    def __init__(self, model):
        self.model = model

        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root)
        self.result_label = self.canvas.create_text(200, 100, text="Nothing drawn yet", tags='gui')

        self.init_graphics()
        self.last_x = 0
        self.last_y = 0
        self.min_x = 0
        self.min_y = 0
        self.max_x = 0
        self.max_y = 0
        self.NN_input = None
        self.image = None
        self.blown_NN_input = None
        self.line_width = 20
        self.update()

    def init_graphics(self):
        self.root.geometry("800x800")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.canvas.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        self.canvas.bind("<Button-1>", self.xy)
        self.canvas.bind("<B1-Motion>", self.add_line)
        self.canvas.bind("<ButtonRelease-1>", lambda event: self.evaluate())
        self.root.bind("+", lambda event: self.line_width_increase())
        self.root.bind("-", lambda event: self.line_width_decrease())
        self.root.bind("<Escape>", lambda event: self.end_program())
        self.root.bind("<Return>", lambda event: self.show_internals())
        self.root.bind("<BackSpace>", lambda event: self.delete_current_drawing())

    def xy(self, event):
        """Takes the coordinates of the mouse when you click the mouse"""
        self.last_x, self.last_y = event.x, event.y

    def add_line(self, event):
        """Creates a line when you drag the mouse
        from the point where you clicked the mouse to where the mouse is now"""
        self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, width=self.line_width, tags='drawing')
        # this makes the new starting point of the drawing
        self.last_x, self.last_y = event.x, event.y

    def update(self):
        self.get_pil_image()
        self.find_number_rectangle()
        if self.max_x != 0:
            self.input_transform()
            width = self.max_x - self.min_x
            height = self.max_y - self.min_y
            im = self.NN_input.resize((int(width*1.4), int(height*1.4)))
            self.blown_NN_input = ImageTk.PhotoImage(im)

    def evaluate(self):
        """Uses the NN to guess the currently drawn number and updates the result_label accordingly"""
        # feed image into NN
        self.update()
        array = np.asarray(self.NN_input)
        array = 255 - array  # invert image - Pillow and MNIST don't agree whether 0 is white or black
        fake_batch = np.array([array])
        prediction = model.predict(fake_batch)
        # update label to show new recognised number
        result = np.argmax(prediction)
        certainty = 100 * np.max(prediction)
        certainty_string = format(certainty, '.2f')
        self.canvas.itemconfig(self.result_label, text=('' + str(result) + ' (' + str(certainty_string) + '%)'))
        print("result: ", result)
        print("certainty: ", certainty, "%")

    def input_transform(self):
        x_size = self.max_x - self.min_x + 1  # both ends inclusive => +1
        y_size = self.max_y - self.min_y + 1
        # crop number down to 20x<20 or <20x20, whichever is possible while conserving aspect ratio
        im = self.image.crop((self.min_x, self.min_y, self.max_x, self.max_y))
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
        self.NN_input = im

    def get_pil_image(self):
        gui_ids = self.canvas.find_withtag("gui")
        self.canvas.itemconfig(gui_ids, fill="white")
        postscript = self.canvas.postscript(colormode='gray')
        self.canvas.itemconfig(gui_ids, fill="black")
        im = Image.open(io.BytesIO(postscript.encode('utf-8')))
        im = im.convert(mode='L')
        self.image = im

    def find_number_rectangle(self):
        array = np.asarray(self.image)
        # find section with number
        array = array - 255  # 255 == code for white
        rows, cols = np.nonzero(array)
        if rows.shape != (0,):
            self.min_x = cols.min()
            self.max_x = cols.max()
            self.min_y = rows.min()
            self.max_y = rows.max()

    def line_width_increase(self):
        """increases line width by 5"""
        self.line_width += 5

    def line_width_decrease(self):
        """decreases line width by 5, but never below 5"""
        if self.line_width > 5:
            self.line_width -= 5

    def delete_current_drawing(self):
        """deletes everything tagged 'drawing'"""
        self.canvas.delete('drawing')
        print("delete_current_drawing")

    def end_program(self):
        """ends the program"""
        exit(0)

    def show_internals(self):
        print("show_internals")
        self.canvas.create_image(self.min_x, self.min_y, anchor=NW, image=self.blown_NN_input, tags="drawing")

    def start(self):
        self.root.mainloop()


gui = Gui(model)
gui.start()

exit(0)
