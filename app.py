from tkinter import *
from PIL import ImageGrab, Image
import numpy as np
from keras.models import load_model

def clear_widget():
    global canvas
    canvas.delete("all")

def activate_event(event):
    global lastx, lasty
    canvas.bind("<B1-Motion>", draw_lines)
    lastx, lasty = event.x, event.y

def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y

    canvas.create_line((lastx, lasty, x, y), width=16, fill="white", capstyle=ROUND, smooth=TRUE, splinesteps=12)

    lastx, lasty = x, y

def screenshot():
    fileName = "prediction.png"
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()

    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()

    ImageGrab.grab().crop((x, y, x1, y1)).save(fileName)

def predict():
    screenshot()
    img = Image.open("prediction.png")
    #resize image to 28x28 pixels
    img = img.resize((28,28))
    #convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    #reshaping to support our model input and normalizing
    img = img.reshape(1,28,28,1)
    img = img/255.0
    #predicting the class
    res = model.predict(img)[0]
    print(np.argmax(res), max(res))
    return np.argmax(res), max(res)

def recognize():
    prediction, acc = predict()
    label.configure(text = str(prediction) + ', ' + str(int(acc * 100)) + '%')

#Model
model = load_model("MNIST.h5")
#App
root = Tk()
root.resizable(0,0)
root.title("Digit Recognition")

#Variables
lastx, lasty = None, None

#Create a canvas for drawing
canvas = Canvas(root, width=500, height=500, background="black")
canvas.grid(row=0, column=0, pady=2, sticky=W)

#Events
canvas.bind("<Button-1>", activate_event)

#Labels
label = Label(text="Thinking..", font=("Helvetica", 48))
label.grid(row=0, column=1, pady=2, padx=2)

#Buttons
button_recognize = Button(text = "Recognize", command = recognize)
button_recognize.grid(row=1, column=1, pady=2, padx=2)

button_clear = Button(text = "Clear", command = clear_widget)
button_clear.grid(row=1, column=0, pady=2)

#Run app
root.mainloop()