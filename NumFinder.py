# Import the necessary libraries
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
import numpy as np


# Function to load the training images and labels
def load_data_from_filenames(directory, image_size):
    images = []
    labels = []

    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Load image
            img_path = os.path.join(directory, filename)
            img = load_img(img_path, target_size=image_size)
            img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
            images.append(img_array)

            # Extract label from filename
            label = int(filename[6])  # Assumes the 7th character is the label
            labels.append(label)

    return np.array(images), np.array(labels)


# Load the training data
data_dir = "C:/python_work/AI/DWG2CAD/TrainingData/Numbers/Basics"
image_size = (50, 50)
x, y = load_data_from_filenames(data_dir, image_size)
x_test, y_test = load_data_from_filenames(data_dir, image_size)

# Split into training and testing
# For the sake of this example, we will use the same data for training and testing
"""
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]"""

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(50, 50, 3)),  # Flatten the input
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')   # Output layer for digits 0-9
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x, y, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Predict
predictions = model.predict(x_test)
print(predictions)