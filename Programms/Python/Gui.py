import io
import math
import tkinter as tk
from tkinter import NW, NE

import numpy as np
from PIL import Image, ImageTk
from scipy import ndimage


def end_program():
    """ends the program"""
    exit(0)


class Gui:
    def __init__(self):
        self.models = []
        self.labels = []

        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root)

        self.init_graphics()
        self.last_x = 0
        self.last_y = 0
        self.min_x = 0
        self.min_y = 0
        self.max_x = 0
        self.max_y = 0
        self.NN_input = None
        self.image = None
        self.line_width = 20
        self.showing_internals = False

    def add_model(self, model):
        self.models.append(model)
        self.canvas.delete("gui")
        self.labels = []
        for i in range(len(self.models)):
            self.labels.append(self.canvas.create_text(10, 100 + i * 50, anchor=NW, text=self.models[i].name + ":/",
                                                       tags="gui"))

    def init_graphics(self):
        """creates the window and key/mouse binds"""
        self.root.geometry("800x800")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.canvas.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        self.canvas.bind("<Button-1>", self.xy)
        self.canvas.bind("<B1-Motion>", self.add_line)
        self.canvas.bind("<ButtonRelease-1>", lambda event: self.evaluate())
        self.root.bind("+", lambda event: self.line_width_increase())
        self.root.bind("-", lambda event: self.line_width_decrease())
        self.root.bind("<Escape>", lambda event: end_program())
        self.root.bind("<Return>", lambda event: self.show_internals())
        self.root.bind("<BackSpace>", lambda event: self.delete_current_drawing())
        print("Keybinds:\nDrawing: drag with left mouse click\nIn/Decrease linewidth: +/-\nDelete drawing: "
              "Backspace\nShow internal picture: Enter\nEnd program: Escape")

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
        """updates the various saved images"""
        self.get_pil_image()
        self.find_number_rectangle()
        if self.max_x != 0:  # has something been drawn?
            self.input_transform()
        else:
            self.NN_input = Image.new("L", (28, 28), color=255)

    def evaluate(self):
        """Uses the NN to guess the currently drawn number and updates the result_label accordingly"""
        # feed image into NN
        self.update()
        array = np.asarray(self.NN_input)
        array = 255 - array  # invert image - Pillow and MNIST don't agree whether 0 is white or black
        predictions = []
        numbers = np.zeros(10)
        most_frequent = 0
        for i in range(len(self.models)):
            predicted = self.models[i].predict(array)
            predictions.append(predicted)
            numbers[predicted] += 1
            if numbers[predicted] > numbers[most_frequent]:
                most_frequent = predicted

        # update labels to show new recognised number
        for i in range(len(self.models)):
            if predictions[i] == most_frequent:
                self.canvas.itemconfig(self.labels[i], text=self.models[i].name + ": " + str(most_frequent),
                                       fill="green")
            else:
                self.canvas.itemconfig(self.labels[i], text=self.models[i].name + ": " + str(predictions[i]),
                                       fill="red")

    def input_transform(self):
        """updates self.NN_input and self.blown_NN_input from self.image"""
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
        """updates self.image to the current drawing on the canvas"""
        gui_ids = self.canvas.find_withtag("gui")
        self.canvas.delete("generated")
        for i in range(len(gui_ids)):
            self.canvas.itemconfig(gui_ids[i], fill="white")
        postscript = self.canvas.postscript(colormode='gray')
        # self.canvas.itemconfig(gui_ids, fill="black")
        im = Image.open(io.BytesIO(postscript.encode('utf-8')))
        im = im.convert(mode='L')
        self.image = im

    def find_number_rectangle(self):
        """updates self.min/max_x/y"""
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
        """deletes everything tagged 'drawing' or 'generated"""
        self.canvas.delete('drawing')
        self.canvas.delete('generated')
        self.showing_internals = False

    def show_internals(self):
        if self.showing_internals:
            self.canvas.delete("generated")
            self.showing_internals = False
        else:
            self.canvas.delete("generated")
            self.canvas.internals_image = ImageTk.PhotoImage(self.NN_input.resize((400, 400)))
            self.canvas.create_image(self.canvas.winfo_width() - 100, 100, anchor=NE, image=self.canvas.internals_image,
                                     tags="generated")
            self.showing_internals = True

    def start(self):
        """starts the tkinter mainloop"""
        self.root.mainloop()
