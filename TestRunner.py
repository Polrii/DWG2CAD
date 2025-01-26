import tensorflow as tf
from PIL import Image
import numpy as np


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
        img = Image.open(image_path).convert("L")  # Convert image to grayscale (L mode)
        
        # Resize the image to specified size
        img = img.resize((50, 50))
        
        # Convert image to NumPy array
        img_array = np.array(img)

        # Normalize pixel values to range 0 to 1
        normalized_array = img_array / 255.0

        return normalized_array

    except Exception as e:
        print(f"Error: {e}")
        return None


# Function to load the model and the desired image to test the model
def ask(model_path, img_path):
    # Load the saved model
    model = tf.keras.models.load_model(model_path)

    # Load the image as an array
    normalized_array = image_to_normalized_array(img_path)
    
    if normalized_array is not None:
        # Reshape the array to match the input shape expected by the model
        normalized_array = np.expand_dims(normalized_array, axis=0)
        
        # Use the model for predictions
        predictions = model.predict(normalized_array)
        return predictions
    else:
        print("Failed to load and process the image.")


# Function to format the predictions
def format_predictions(predictions):
    for prediction in predictions:
        for x, answer in enumerate(prediction):
            print(f"Number {x}: {answer}")


# Run the script
if __name__ == "__main__":
    model_path = "C:/python_work/AI/DWG2CAD/Models/NumFinderFonts.keras"
    img_path = "C:/python_work/AI/DWG2CAD/TrainingData/Numbers/Fonts/Number9 (3).png"
    predictions = ask(model_path, img_path)
    format_predictions(predictions)
    
    