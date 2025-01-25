import tensorflow as tf
import numpy as np

# Generate dummy data
X = np.random.rand(1000, 20)  # 1000 samples, 20 features each
y = np.random.randint(0, 2, size=(1000,))  # Binary classification labels

# Split into training and testing
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(20,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Predict
predictions = model.predict(X_test)
