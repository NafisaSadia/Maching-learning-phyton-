
# 🧠 Fetal Brain Abnormality Detection (Ultrasound) - CNN in Google Colab

## 📦 Step 1: Install and Load Dataset from Roboflow
!pip install roboflow --quiet

from roboflow import Roboflow
rf = Roboflow(api_key="R8eB6vjWYvWSa9nX7A3r")  # Your actual API key
project = rf.workspace("hritwik-trivedi-gkgrv").project("fetal-brain-abnormalities-ultrasound")
dataset = project.version(1).download("folder")

## 📁 Step 2: Load and Prepare Data
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Parameters
img_size = 128
data_dir = dataset.location + "/train"
categories = os.listdir(data_dir)

# Load images and labels
X, y = [], []
for label, category in enumerate(categories):
    folder = os.path.join(data_dir, category)
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            X.append(img)
            y.append(label)

X = np.array(X) / 255.0  # Normalize
y = to_categorical(np.array(y))  # One-hot encode labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## 🧠 Step 3: Define a Simple CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

## 🚀 Step 4: Train the Model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

## 📊 Step 5: Evaluate and Predict
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()

# Show predictions
predictions = model.predict(X_test[:5])
for i, pred in enumerate(predictions):
    plt.imshow(X_test[i])
    plt.title(f"Predicted: {categories[np.argmax(pred)]}, Actual: {categories[np.argmax(y_test[i])]}")
    plt.axis("off")
    plt.show()
# Maching-learning-phyton-
