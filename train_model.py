import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

data = pd.read_csv("fer2013.csv")

# Extracting features and labels
X = []
y = []
for index, row in data.iterrows():
    pixels = np.array(row["pixels"].split(), dtype="float32").reshape(48, 48)
    X.append(pixels)
    y.append(row["emotion"])

# Converting to numpy arrays
X = np.array(X)
y = np.array(y)

# Normalizing the pixel values 0 to 1
X = X / 255.0

# Reshaping for CNN input or simply adding channel dimension
X = X.reshape(-1, 48, 48, 1)

# One-hot encoding labels
y = to_categorical(y, num_classes=7)

# Splitting datasets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  
])

# Compiling 
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Training
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_val, y_val))


model.save("emotion_model.h5")

# Plot training history
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.show()
