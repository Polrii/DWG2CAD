from tkinter import *
from tkinter import filedialog
import customtkinter
import random


# Function to select a file
def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        print(f"Selected file: {file_path}")
        
        
# Function to generate random bars
def generate_bars():
    canvas.delete("all")  # Clear the canvas
    bar_width = 30
    spacing = 10
    start_x = 20

    for i in range(10):
        random_value = random.random()  # Generate a random number between 0 and 1

        # Calculate bar dimensions
        x0 = start_x + i * (bar_width + spacing)
        y0 = 340  # Bottom of the canvas
        x1 = x0 + bar_width
        y1 = 340 - (random_value * 320)  # Scale height to canvas

        # Draw the bar
        canvas.create_rectangle(x0, y0, x1, y1, fill="blue")

        # Add the value as text above the bar
        canvas.create_text((x0 + x1) / 2, y1 - 10, text=f"{random_value:.2f}", fill="black")
        
        
# Initialize the root window
root = customtkinter.CTk()
root.title("AI Model Tester")
root.geometry("800x600")

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
select_model_button = customtkinter.CTkButton(master=model_frame, text="Select Model", command=select_file)
select_model_button.pack(side=LEFT, anchor=NW, padx=16, pady=16)

# Create a text box to display the image file path
image_textbox = customtkinter.CTkTextbox(master=image_frame, width=600, height=1)
image_textbox.insert("0.0", "NO IMAGE SELECTED") # Insert at line 0, character 0
image_textbox.configure(state="disabled")  # Disable textbox to be read-only
image_textbox.pack(side=LEFT, anchor=NW, padx=16, pady=16)

# Create a button to select an image
select_image_button = customtkinter.CTkButton(master=image_frame, text="Select Image", command=select_file)
select_image_button.pack(side=LEFT, anchor=NW, padx=16, pady=16)

# Create a button to run the model
run_model_button = customtkinter.CTkButton(master=root, text="Run Model", command=generate_bars)
run_model_button.pack(padx=16, pady=16)


# Canvas to draw bars
canvas = customtkinter.CTkCanvas(root, width=430, height=360, bg="white")
canvas.pack(pady=10)


root.mainloop()
