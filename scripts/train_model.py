# --------------------------------------------------------------------------------------------
#
# Title: Train Model (Project Part 2)
# Course: CSCI 3240 (Computational Photography)
# Authors: Amit Sarvate and Nirujan Velvarathan
# Date: December 8th 2024
#
# Description: The purpose of this script is to allow us train the classification model using 
# the data that we collected in Part 1 in order to use the model for the next part of project 
#
# --------------------------------------------------------------------------------------------

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import os

# Ensure the 'models' folder exists, create it if it doesn't
os.makedirs("models", exist_ok=True)

# Load the dataset from the CSV file
data = np.loadtxt("data/gesture_data.csv", delimiter=",")
X = data[:, :-1]  # Landmark coordinates (Features)
y = data[:, -1]   # Gesture labels (Target)

# Adjust labels to start from 0
y = y - 1  # Subtract 1 from all label values

# Normalize data
X -= X.min(axis=0)
X /= X.max(axis=0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define number of classes
num_classes = len(np.unique(y))

# Define the neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')  # Adjust output layer
])

# Compile the model with optimizer, loss function, and evaluation metric
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model using the training data
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model on the testing data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

# Save the trained model to the 'models' folder
model.save("models/gesture_recognition_model.h5")
