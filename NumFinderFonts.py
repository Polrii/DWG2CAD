# Import the necessary libraries
import os
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
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


# Function to load the training images and labels
def load_data_from_filenames(directory):
    images = []
    labels = []

    for i in range(10):
        for filename in os.listdir(directory):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                # Load image
                img_path = os.path.join(directory, filename)
                normalized_array = image_to_normalized_array(img_path)
                images.append(normalized_array)

                # Extract label from filename
                label = int(filename[6])  # Assumes the 7th character is the label
                labels.append(label)

    return np.array(images), np.array(labels)


# Load the training data
data_dir = "C:/python_work/AI/DWG2CAD/TrainingData/Numbers/Fonts"
x, y = load_data_from_filenames(data_dir)

# Ensure data has correct dimensions
x = np.expand_dims(x, axis=-1)  # Add channel dimension for grayscale images

# Split into training and testing
x_train, x_test = x[:600], x[600:]
y_train, y_test = y[:600], y[600:]

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(50, 50, 1)),  # Define the input shape explicitly
    tf.keras.layers.Flatten(),                # Flatten the input
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')   # Output layer for digits 0-9
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Predict
predictions = model.predict(x_test)
#print(predictions)

# Save the model in the SavedModel format
model.save("C:/python_work/AI/DWG2CAD/Models/NumFinderFonts.keras")