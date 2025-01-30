import tensorflow as tf
import PIL
import numpy as np
from tkinter import *
from tkinter import filedialog
import customtkinter


# Define the global variables
global model_path_text
global img_path_text


# Function to format the image
def image_to_normalized_array(image_path):
    """
    Opens an image, converts it to grayscale, and normalizes pixel values.
    Args:
        image_path (str): The path to the image file.
    Returns:
        np.ndarray: A NumPy array with normalized pixel values (0 to 1).
    """
    try:
        # Open the image using Pillow
        img = PIL.Image.open(image_path).convert("L")  # Convert image to grayscale (L mode)
        
        # Resize the image to specified size
        img = img.resize((50, 50))
        
        # Convert image to NumPy array
        img_array = np.array(img)

        # Normalize pixel values to range 0 to 1
        normalized_array = img_array / 255.0
        
        # Reshape the array to match the input shape expected by the model
        normalized_array = np.expand_dims(normalized_array, axis=0)

        return normalized_array

    except Exception as e:
        print(f"Error: {e}")
        return None


# Function to load the model and the desired image to test the model
def ask():
    global model_path_text
    global img_path_text
    
    # Load the saved model
    model = tf.keras.models.load_model(model_path_text)
    
    # Iterate across the images
    predictions = []
    images_array = []
    for image_path in img_path_text:
        # Load the image as an array
        normalized_array = image_to_normalized_array(image_path)
        
        if normalized_array is not None:
            images_array.append(normalized_array)
        else:
            print("Failed to load and process the image.")
    
    # Use the model for predictions
    predictions = model.predict(np.array(images_array))
    return predictions


# Function to select a file
def select_file(element):
    global model_path_text
    global img_path_text
    
    if element == "model":
        file_path = filedialog.askopenfilename()
        if file_path:
            model_textbox.configure(state="normal")
            model_textbox.delete("0.0", "end")
            model_textbox.insert("0.0", file_path)
            model_textbox.configure(state="disabled")
            model_path_text = file_path
        
    elif element == "image":
        file_paths = filedialog.askopenfilenames()
        if file_paths:
            image_textbox.configure(state="normal")
            image_textbox.delete("0.0", "end")
            image_textbox.insert("0.0", "\n".join(file_paths))
            image_textbox.configure(state="disabled")
            img_path_text = file_paths


# Function to generate random bars
def generate_bars():
    canvas.delete("all")  # Clear the canvas
    bar_width = 30
    spacing = 10
    start_x = 20
    
    # Create a list with the numbers of each bar
    numbers = [0] * 10
    predictions = ask()
    length = len(predictions)
    for prediction in predictions:
        for i, answer in enumerate(prediction):
            numbers[i] += answer
            
    # Draw the bars
    for i, number in enumerate(numbers):
        # Calculate bar dimensions
        x0 = start_x + i * (bar_width + spacing)
        y0 = 340  # Bottom of the canvas
        x1 = x0 + bar_width
        y1 = 340 - (number / length * 320)  # Scale height to canvas
        
        # Draw the bar
        canvas.create_rectangle(x0, y0, x1, y1, fill="blue")

        # Add the value as text above the bar
        if number > 0.01:
            canvas.create_text((x0 + x1) / 2, y1 - 10, text=f"{number:.2f}", fill="black")
        else:
            canvas.create_text((x0 + x1) / 2, y1 - 10, text=f"{number:.0e}", fill="black")
        
        # Add the element as text below the bar
        canvas.create_text((x0 + x1) / 2, y0 + 10, text=f"{i}", fill="black")


# Initialize the root window
root = customtkinter.CTk()
root.title("AI Model Tester")
root.geometry("800x700")

# Create a frame to hold the text boxes and buttons for the model selection
model_frame = customtkinter.CTkFrame(master=root, fg_color="transparent")
model_frame.pack(side=TOP, fill=X, padx = 8)

# Create a frame to hold the text boxes and buttons for the image selection
image_frame = customtkinter.CTkFrame(master=root, fg_color="transparent")
image_frame.pack(side=TOP, fill=X, padx = 8)


# Create a text box to display the model file path
model_textbox = customtkinter.CTkTextbox(master=model_frame, width=600, height=1)
model_textbox.insert("0.0", "NO FILE SELECTED") # Insert at line 0, character 0
model_textbox.configure(state="disabled")  # Disable textbox to be read-only
model_textbox.pack(side=LEFT, anchor=NW, padx=16, pady=16)

# Create a button to select a model
select_model_button = customtkinter.CTkButton(master=model_frame, text="Select Model", command=lambda: select_file("model"))
select_model_button.pack(side=LEFT, anchor=NW, padx=16, pady=16)

# Create a text box to display the image file path
image_textbox = customtkinter.CTkTextbox(master=image_frame, width=600, height=100)
image_textbox.insert("0.0", "NO IMAGES SELECTED") # Insert at line 0, character 0
image_textbox.configure(state="disabled")  # Disable textbox to be read-only
image_textbox.pack(side=LEFT, anchor=NW, padx=16, pady=16)

# Create a button to select an image
select_image_button = customtkinter.CTkButton(master=image_frame, text="Select Images", command=lambda: select_file("image"))
select_image_button.pack(side=LEFT, anchor=NW, padx=16, pady=16)

# Create a button to run the model
run_model_button = customtkinter.CTkButton(master=root, text="Run Model", command=generate_bars)
run_model_button.pack(padx=16, pady=16)

# Canvas to draw bars
canvas = customtkinter.CTkCanvas(root, width=430, height=360, bg="white")
canvas.pack(pady=10)


root.mainloop()
