import os
import cv2
import uuid
import argparse
import numpy as np
from glob import glob


def show_images(original, boxes, annotated):
    cv2.imshow("original", original)
    cv2.moveWindow("original", 0, 0)
    cv2.imshow("boxes", boxes)
    cv2.moveWindow("boxes", 0, 125)
    cv2.imshow("annotated", annotated)
    cv2.moveWindow("annotated", 0, 250)


def get_images():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i", "--input", required=True, help="path to input directory of images"
    )
    ap.add_argument(
        "-o", "--output", required=True, help="path to output directory of annotations"
    )
    args = vars(ap.parse_args())
    input_files = glob(f"{args['input']}/*.jpg")
    random_files = np.random.choice(input_files, size=(50,), replace=False)
    return random_files


def morphological_operations(original_image):
    gaussian = cv2.GaussianBlur(original_image, (5, 5), 0)
    _, thresholded = cv2.threshold(gaussian, 113, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(thresholded, kernel, iterations=1)
    kernel = np.ones((4, 2), np.uint8)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    # kernel = np.ones((3, 3), np.uint8)
    # opening = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((2, 2), np.uint8)
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    return closing


def remove_small_contours(contours, min_area):
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    return filtered_contours


def save_roi_image(roi, folder):
    os.makedirs(f"./annotations/{folder}", exist_ok=True)
    cv2.imwrite(f"./annotations/{folder}/{uuid.uuid4()}.jpg", roi)
    print(f"Image saved to folder './annotations/{folder}'.")

if __name__ == "__main__":
    files = get_images()
    for image_path in files:
        image = cv2.imread(image_path)
        image_boxes = image.copy()
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        altered_image = morphological_operations(gray_image)
        contours, _ = cv2.findContours(altered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        
        # Remove small contours
        min_area = 90
        filtered_contours = remove_small_contours(sorted_contours, min_area)
        
        mask = np.zeros(altered_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, filtered_contours, -1, 255, -1)
        result = cv2.bitwise_and(altered_image, altered_image, mask=mask)
        
        for c in filtered_contours:
            x, y, w, h = cv2.boundingRect(c)
            roi = result[y: y + h, x: x + w]
            if h > 10:
                cv2.rectangle(image_boxes, (x, y), (x + w, y + h), (0, 255, 0), 1)
                show_images(image, image_boxes, result)
                
                if w > 35:  # 2 connected letters are split evenly
                    mid = int(roi.shape[1] / 2)
                    roi1 = roi[0: roi.shape[0], 0:mid]
                    roi2 = roi[0: roi.shape[0], mid: mid + roi.shape[1]]
                    letters = [roi1, roi2]
                    cv2.imshow("Left Crop", roi1)
                    cv2.imshow("Right Crop", roi2)
                    cv2.moveWindow("Left Crop", 275, 0)
                    cv2.moveWindow("Right Crop", 275, 75)
                    
                    for roi in letters:
                        res = cv2.waitKey(0)
                        save_roi_image(roi, chr(res % 256))
                        
                    cv2.destroyWindow("Left Crop")
                    cv2.destroyWindow("Right Crop")
                else:
                    cv2.imshow("Single Crop", roi)
                    cv2.moveWindow("Single Crop", 275, 0)
                    
                    res = cv2.waitKey(0)
                    save_roi_image(roi, chr(res % 256))
                    
                    cv2.destroyWindow("Single Crop")
        
        cv2.destroyAllWindows()