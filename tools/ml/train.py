# python train_model.py --dataset dataset --model models/model_2_may_30_2024.keras --plot plots/model_2_may_30_2024.png

import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Parse command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# Set image dimensions and other parameters
image_size = (28, 28)
num_classes = 26
batch_size = 32
epochs = 10

# Load and preprocess the dataset
data = []
labels = []

for label in os.listdir(args["dataset"]):
    label_dir = os.path.join(args["dataset"], label)
    for image_file in os.listdir(label_dir):
        image_path = os.path.join(label_dir, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, image_size)
        image = image.astype("float32") / 255.0  # Normalize pixel values
        data.append(image)
        labels.append(label)

data = np.array(data)
labels = np.array(labels)

# Split the data into training and testing sets
(train_data, test_data, train_labels, test_labels) = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# One-hot encode the labels
lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
test_labels = lb.transform(test_labels)

# Reshape the data for the CNN
train_data = train_data.reshape((train_data.shape[0], 28, 28, 1))
test_data = test_data.reshape((test_data.shape[0], 28, 28, 1))

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Train the model
history = model.fit(
    train_data,
    train_labels,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(test_data, test_labels),
)

# Save the trained model
model.save(args["model"])

# Plot the training and validation accuracy and loss
plt.style.use("ggplot")
plt.figure()
plt.plot(history.history["accuracy"], label="train_accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Training Accuracy and Loss")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy/Loss")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
