# Import necessary libraries
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
import cv2

# Load your trained model
model = load_model('trafficsign.h5')

# Define the classes (Traffic Sign Names)
classes = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

# Initialize the tkinter window
window = tk.Tk()
window.title("Traffic Sign Classifier")
window.geometry("600x400")

# Create a label to display the traffic sign name
label = tk.Label(window, text="Select an image to classify the traffic sign", font=("Arial", 16))
label.pack(pady=20)

# Create a label to display the image
image_label = tk.Label(window)
image_label.pack()

# Function to load and display the image
def load_image():
    global img_path
    img_path = filedialog.askopenfilename()
    img = Image.open(img_path)
    img = img.resize((150, 150))  # Resize for display
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk
    label.config(text="Image loaded. Press 'Predict' to classify the sign.")

# Function to predict the traffic sign
def predict_sign():
    # Preprocess the image for the model
    img = cv2.imread(img_path)
    img = cv2.resize(img, (30, 30))  # Resize to the required input shape
    img = np.expand_dims(img, axis=0)
    
    # Predict the traffic sign
    pred = model.predict(img)
    sign_index = np.argmax(pred)
    
    # Display the result
    sign_name = classes[sign_index]
    label.config(text=f"Prediction: {sign_name}")

# Function to reset the interface
def reset_interface():
    image_label.config(image='')
    label.config(text="Select an image to classify the traffic sign")

# Buttons for loading, predicting, and resetting
load_button = tk.Button(window, text="Load Image", command=load_image, font=("Arial", 14))
load_button.pack(pady=10)

predict_button = tk.Button(window, text="Predict", command=predict_sign, font=("Arial", 14))
predict_button.pack(pady=10)

reset_button = tk.Button(window, text="Reset", command=reset_interface, font=("Arial", 14))
reset_button.pack(pady=10)

# Run the Tkinter window
window.mainloop()
