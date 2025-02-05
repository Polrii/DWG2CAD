"""# Basic version
from ultralytics import YOLO

# Load the trained model
model = YOLO("C:/python_work/AI/DWG2CAD/runs/detect/train2/weights/best.pt")  # Or "last.pt" for the latest state


results = model("C:/python_work/AI/DWG2CAD/TrainingData/NumsDataset.v5i.yolov8/test/images/24-01-01_drawing_jpg.rf.1524bae88e98b40ba874f3371cd2f9f6.jpg")  # Run object detection

# Show the detected numbers
results[0].show()  # Displays the image with bounding boxes
"""


"""
Add a checkbox to choose if you want to save the images or not
"""


from PIL import Image
from tkinter import *
from tkinter import filedialog
import customtkinter
from ultralytics import YOLO


# Define the global variables
global model_path_text
global img_path_text


# Function to load the model and the desired image to test the model
def ask():
    global model_path_text
    global img_path_text
    
    # Load the saved model
    model = YOLO(model_path_text)
    
    # Use the model
    results = model(img_path_text, save=False)  # Run object detection

    return results


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
def execute_model():
    canvas.delete("all")  # Clear the canvas
    
    # Get the predictions
    results = ask()
    for result in results:
        result.show()

    
    my_image = customtkinter.CTkImage(light_image=Image.open("<path to light mode image>"),
                                  dark_image=Image.open("<path to dark mode image>"),
                                  size=(30, 30))

    image_label = customtkinter.CTkLabel(root, image=my_image, text="")  # display image with a CTkLabel
    

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
run_model_button = customtkinter.CTkButton(master=root, text="Run Model", command=execute_model)
run_model_button.pack(padx=16, pady=16)

# Canvas to draw bars
canvas = customtkinter.CTkCanvas(root, width=430, height=360, bg="white")
canvas.pack(pady=10)


root.mainloop()