# python test_model.py --model models/model.keras --image captcha.jpg

import argparse
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer


def morphological_operations(original_image):
    # Apply Gaussian blur to reduce noise
    gaussian = cv2.GaussianBlur(original_image, (5, 5), 0)

    # Threshold the image to create a binary image
    _, thresholded = cv2.threshold(gaussian, 113, 255, cv2.THRESH_BINARY_INV)

    # Apply erosion to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(thresholded, kernel, iterations=1)

    # Apply dilation to connect broken components
    kernel = np.ones((4, 2), np.uint8)
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    # Apply morphological closing to fill small gaps and holes
    kernel = np.ones((2, 2), np.uint8)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    return closed


def remove_small_contours(contours, min_area):
    # Filter out contours with area smaller than the specified minimum area
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    return filtered_contours


def resize_roi(image, image_size):
    # Resize the image to the specified size and normalize pixel values
    resized_image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA) / 255.0
    resized_image = resized_image.reshape((1, 28, 28, 1))
    return resized_image


# Parse command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained model")
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# Set image dimensions
image_size = (28, 28)

# Load the trained model
model = load_model(args["model"])

# Load and preprocess the input image
image = cv2.imread(args["image"], cv2.IMREAD_GRAYSCALE)
processed_image = morphological_operations(image)
contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

# Remove small contours
min_area = 90
filtered_contours = remove_small_contours(sorted_contours, min_area)

if len(filtered_contours) == 0:
    print("No contours found in the image.")
    exit()

mask = np.zeros(processed_image.shape, dtype=np.uint8)
cv2.drawContours(mask, filtered_contours, -1, 255, -1)
masked_image = cv2.bitwise_and(processed_image, processed_image, mask=mask)

# Initialize LabelBinarizer
lb = LabelBinarizer()
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
lb.fit(labels)

word = []
for c in filtered_contours:
    x, y, w, h = cv2.boundingRect(c)
    roi = masked_image[y: y + h, x: x + w]
    if h > 10:
        if w > 35:  # 2 connected letters are split evenly
            mid = int(roi.shape[1] / 2)
            roi1 = roi[0: roi.shape[0], 0:mid]
            roi2 = roi[0: roi.shape[0], mid: mid + roi.shape[1]]
            letters = [roi1, roi2]
            for letter in letters:
                letter = resize_roi(letter, image_size)
                predictions = model.predict(letter)
                predicted_label = lb.classes_[predictions.argmax()]
                word.append(predicted_label)
        else:
            letter = resize_roi(roi, image_size)
            predictions = model.predict(letter)
            predicted_label = lb.classes_[predictions.argmax()]
            word.append(predicted_label)

# Print the predicted label
print("Predicted Label:", ''.join(word))
