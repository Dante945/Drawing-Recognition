from tkinter import * # UI library
from keras.models import load_model # Imports the trained models
import numpy as np # Resizing
#import pywin32
import win32gui # Libraries for sending the drawn image on the U.I to the trained model
#from pillow import ImageGrab, ImageOps
from PIL import ImageGrab, ImageOps

# Importing both digit and clothing models
digit_model = load_model('Digit_Weights.h5')
clothing_model = load_model('Clothing_Weights.h5')

# Global variables for brush propreties
BRUSH_THICKNESS = 8
BRUSH_PERCENT = 50

# Maps predicted category (0-9) to clothing label
clothing_converter = {
    9: "Boot",
    8: "Bag",
    7: "Sneaker",
    6: "Shirt",
    5: "Sandal",
    4: "Coat",
    3: "Dress",
    2: "Pull-over",
    1: "Trousers",
    0: "T-Shirt",

}


#---------------------------- Functions ------------------------------- #
def clear_canvas():
    canvas.delete("all")


# Displays results and percentage on screen
def update_canvas(x, y, is_digit):
    if is_digit:
        result_label.config(text=f"Result:{x}")
    else:
        result_label.config(text=f"Result:{clothing_converter[x]}")

    confidence_label.config(text=f"Confidence:{int(y*100)}%")


def predict(drawing, choice):
    """
    Processing:
    Take the image of the drawing pad and turn it into a 28x28 image
    Convert rgb to gray scale to account for how the model was trained
    Invert the image seeing as the drawing pad inverts the image on its own
    Shape the image to the dimensions of the input for our model and normalize the values
    Chooses which model to use based on which button is pressed
    Sends the image through the model and returns the classified digit along with the accuracy/confidence
    """
    drawing = drawing.resize((28, 28))
    drawing = drawing.convert('L')
    drawing = ImageOps.invert(drawing)
    drawing = np.array(drawing)
    drawing = drawing.reshape(1, 28, 28, 1)
    drawing = drawing / 255.0

    if choice == 0:
        res = clothing_model.predict([drawing])[0]
    else:
        res = digit_model.predict([drawing])[0]
    return np.argmax(res), max(res)


"""
Drawing:
Takes the current x and y coords when you press the left-click button
At these coordinates, it creates a small oval which acts as a paintbrush stroke 
"""
def draw_lines(position):
    x_cord = position.x
    y_cord = position.y

    # Upper left boundary of the oval
    top_x = x_cord + BRUSH_THICKNESS
    top_y = y_cord + BRUSH_THICKNESS

    # Bottom Right boundary of the oval
    bot_x = x_cord - BRUSH_THICKNESS
    bot_y = y_cord - BRUSH_THICKNESS

    canvas.create_oval(bot_x, bot_y, top_x, top_y, fill='black')


"""
Changes the thickness of the brush depending on which button in pressed
Also checks to make sure the thickness doesnt grow to large or small
Reflects current thickness on screen
"""


def increase_thickness():
    global BRUSH_THICKNESS, BRUSH_PERCENT
    if BRUSH_PERCENT < 100:
        BRUSH_PERCENT += 10
        thickness_label.config(text=f"   Thickness: {BRUSH_PERCENT}%")
        BRUSH_THICKNESS += 2


def decrease_thickness():
    global BRUSH_THICKNESS, BRUSH_PERCENT

    if BRUSH_PERCENT > 0:
        BRUSH_PERCENT -= 10
        thickness_label.config(text=f"   Thickness: {BRUSH_PERCENT}%")
        BRUSH_THICKNESS -= 2


# Takes the current marks on the canvas and sends it to the prediction function
def export_canvas_digit():
    canvas_id = canvas.winfo_id()
    drawing_coords = win32gui.GetWindowRect(canvas_id)
    drawing = ImageGrab.grab(drawing_coords)

    digit, confidence = predict(drawing, 1)
    update_canvas(digit, confidence, is_digit=True)


def export_canvas_clothing():
    canvas_id = canvas.winfo_id()
    drawing_coords = win32gui.GetWindowRect(canvas_id)
    drawing = ImageGrab.grab(drawing_coords)

    digit, confidence = predict(drawing, 0)
    update_canvas(digit, confidence, is_digit=False)


# ---------------------------- UI SETUP ------------------------------- #

# Simple Tkinter window set up with buttons, labels, a canvas

# Window
window = Tk()
window.title("Digit Drawing Pad")
window.minsize(600, 600)

# Canvas
canvas = Canvas(bg="white", cursor="pencil", height=500, width=500)
canvas.grid(column=0, row=1, columnspan=5, padx=40, pady=40)
# Clicking left mouse button will create a continuous line of ovals which resembles a brush in paint
canvas.bind("<B1-Motion>", draw_lines)

# Labels
thickness_label = Label(text="   Thickness: 50%", font=("Courier",12, "normal"))
thickness_label.grid(row=3, column=0, columnspan=2)

predict_label = Label(text="Predict!", font=("Courier", 12, "normal"))
predict_label.grid(column=2, row=3, columnspan=2, pady=10)

confidence_label = Label(text="Confidence: ", font=("Courier",16, "normal"))
confidence_label.grid(column=4, row=3, pady =15)

title = Label(text="-Drawing Recognizer-", font=("Courier",32, "bold"))
title.grid(column=0, row=0, columnspan=5, padx=10, pady=10)

result_label = Label(text="Result: ", font=("Courier", 16, "normal"))
result_label.grid(column=4, row=2, pady=15)

# Buttons
clear_button = Button(text="Clear Canvas", command=clear_canvas ,padx= 100)
clear_button.grid(column=0, row=2, columnspan=4, pady=10)

predict_digit_button = Button(text="Digit", command=export_canvas_digit)
predict_digit_button.grid(column=2, row=4, pady=10)

predict_cloth_button = Button(text="Clothing", command=export_canvas_clothing)
predict_cloth_button.grid(column=3, row=4, pady=10)

Thickness_up = Button(text="+", command=increase_thickness)
Thickness_up.grid(column=1, row=4, pady=10)

Thickness_down = Button(text="-", command=decrease_thickness)
Thickness_down.grid(column=0, row=4, pady=10)

window.mainloop()
