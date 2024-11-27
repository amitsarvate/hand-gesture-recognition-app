import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import os

# Ensure the 'models' folder exists
os.makedirs("models", exist_ok=True)

# Load dataset
data = np.loadtxt("data/gesture_data.csv", delimiter=",")
X = data[:, :-1]  # Landmark coordinates
y = data[:, -1]   # Gesture labels

# Check if there are any 0 labels, and adjust them accordingly
if np.any(y == 0):
    print("Warning: Some labels are zero. They will be adjusted.")

# Ensure the labels start from 0 (if needed)
# If labels start from 1, subtract 1 to make them start from 0
# Avoid adjusting if the labels already start from 0
if np.min(y) > 0:
    y = y - 1  # Subtract 1 from all label values to make them start from 0

# Normalize data
X -= X.min(axis=0)
X /= X.max(axis=0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define number of classes
num_classes = len(np.unique(y))

# Define the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')  # Adjust output layer
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

# Save the model
model.save("models/gesture_recognition_model.h5")
