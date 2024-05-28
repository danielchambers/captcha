import cv2
import numpy as np
import requests


def url_to_image(url, headers):
    http_response = requests.get(url, headers=headers).content
    image = np.asarray(bytearray(http_response), dtype="uint8")
    return cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)


def odd_number(number):
    if number % 2 == 0:
        number += 1
    return number


def apply_operations(original_image, gaussian_blur_ksize, threshold_value, erosion_kernel_x, erosion_kernel_y, erosion_iterations, dilation_kernel_x, dilation_kernel_y, dilation_iterations, morph_open_kernel, morph_close_kernel):
    # Apply Gaussian Blur
    ksize = (odd_number(gaussian_blur_ksize), odd_number(gaussian_blur_ksize))
    sigma = 0
    gaussian = cv2.GaussianBlur(original_image, ksize, sigma)

    # Apply thresholding
    _, thresholded = cv2.threshold(gaussian, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Apply erosion
    kernel = np.ones((erosion_kernel_x, erosion_kernel_y), np.uint8)
    eroded = cv2.erode(thresholded, kernel, iterations=erosion_iterations)

    # Apply dilation
    kernel = np.ones((dilation_kernel_x, dilation_kernel_y), np.uint8)
    dilated = cv2.dilate(eroded, kernel, iterations=dilation_iterations)

    # Apply morphology open
    kernel = np.ones((morph_open_kernel, morph_open_kernel), np.uint8)
    opening = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel)

    # Apply morphology close
    kernel = np.ones((morph_close_kernel, morph_close_kernel), np.uint8)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    return closing
