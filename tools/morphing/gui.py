# main_window.py

from PyQt5.QtWidgets import QMainWindow, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from image_utils import url_to_image, apply_operations
from captchas import site


class MainWindow(QMainWindow):
    def __init__(self, captcha_data):
        super().__init__()
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('Morphological Operations')
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Load and display the image
        url = site(captcha_data['name'], random_str=captcha_data['ramdom_str'])
        original_image = url_to_image(url, {"User-Agent": "Mozilla/5.0"})
        self.original_image = original_image
        self.altered_image = original_image.copy()

        # Threshold parameters
        self.threshold_value = 0
        # Gaussian Blur parameters
        self.gaussian_blur_ksize = 1
        # Erosion parameters
        self.erosion_kernel_x = 1
        self.erosion_kernel_y = 1
        self.erosion_iterations = 1
        # dilation parameters
        self.dilation_kernel_x = 1
        self.dilation_kernel_y = 1
        self.dilation_iterations = 1
        # Morphology Open parameters
        self.morph_open_kernel = 1
        # Morphology Close parameters
        self.morph_close_kernel = 1

        # Create a horizontal layout for the images
        images_layout = QHBoxLayout()
        images_layout.setContentsMargins(90, 25, 90, 25)
        images_layout.setAlignment(Qt.AlignCenter)

        # Add original image label
        self.original_image_label = QLabel()
        self.display_image(self.original_image_label, original_image)
        images_layout.addWidget(self.original_image_label)

        # Add altered image label
        self.altered_image_label = QLabel()
        self.display_image(self.altered_image_label, self.altered_image)
        images_layout.addWidget(self.altered_image_label)
        main_layout.addLayout(images_layout)

        main_layout.addSpacing(5)
        self.add_print_button(main_layout)
        main_layout.addSpacing(5)
        self.add_operation_sliders(
            main_layout,
            "Gaussian Blur",
            ["Iterations"],
            10,
            self.gaussian_blur_ksize_changed,
        )
        main_layout.addSpacing(5)
        self.add_operation_sliders(
            main_layout, "Threshold", ["Value"], 255, self.threshold_changed
        )
        main_layout.addSpacing(5)
        self.add_operation_sliders(
            main_layout,
            "Erosion",
            ["Kernel X", "Kernel Y", "Iterations"],
            10,
            self.erosion_kernel_x_changed,
            self.erosion_kernel_y_changed,
            self.erosion_iterations_changed,
        )
        main_layout.addSpacing(5)
        self.add_operation_sliders(
            main_layout,
            "Dilation",
            ["Kernel X", "Kernel Y", "Iterations"],
            10,
            self.dilation_kernel_x_changed,
            self.dilation_kernel_y_changed,
            self.dilation_iterations_changed,
        )
        main_layout.addSpacing(5)
        self.add_operation_sliders(
            main_layout,
            "Morphology Open",
            ["Kernel X", "Kernel Y", "Iterations"],
            10,
            self.morph_open_kernel_changed,
        )
        main_layout.addSpacing(5)
        self.add_operation_sliders(
            main_layout,
            "Morphology Close",
            ["Kernel X", "Kernel Y", "Iterations"],
            10,
            self.morph_close_kernel_changed,
        )
        main_layout.addSpacing(5)

    def display_image(self, label, image):
        height, width = image.shape
        q_image = QImage(image, width, height, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)

    def update_altered_image_display(self, image):
        height, width = image.shape
        q_image = QImage(image, width, height, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        self.altered_image_label.setPixmap(pixmap)

    def add_operation_sliders(self, layout, operation_name, slider_labels, slider_range, *callbacks):
        operation_slider_layout = QVBoxLayout()
        operation_slider_layout.setContentsMargins(25, 0, 25, 40)
        operation_slider_layout.setSpacing(5)
        for label, callback in zip(slider_labels, callbacks):
            self.add_slider(operation_slider_layout, f"{operation_name} {label}", slider_range, callback)
        layout.addLayout(operation_slider_layout)

    def add_slider(self, layout, label_text, slider_range, callback):
        label = QLabel(label_text)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(1, slider_range)
        slider.valueChanged.connect(callback)
        layout.addWidget(label)
        layout.addWidget(slider)
    
    def add_print_button(self, layout):
        print_button = QPushButton("Print Code")
        print_button.clicked.connect(self.print_parameters)
        print_button.setFixedSize(150, 30)
        layout.addWidget(print_button, alignment=Qt.AlignCenter)

    def print_parameters(self):
        print(f"gaussian = cv2.GaussianBlur(original_image, ({self.gaussian_blur_ksize}, {self.gaussian_blur_ksize}), 0)")
        print(f"_, thresholded = cv2.threshold(gaussian, {self.threshold_value}, 255, cv2.THRESH_BINARY_INV)")
        print(f"kernel = np.ones(({self.erosion_kernel_x}, {self.erosion_kernel_y}), np.uint8)")
        print(f"eroded = cv2.erode(thresholded, kernel, iterations={self.erosion_iterations})", self.erosion_kernel_y)
        print(f"kernel = np.ones(({self.dilation_kernel_x}, {self.dilation_kernel_y}), np.uint8)")
        print(f"dilated = cv2.dilate(eroded, kernel, iterations={self.dilation_iterations})")
        print(f"kernel = np.ones(({self.morph_open_kernel}, {self.morph_open_kernel}), np.uint8)")
        print(f"opening = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel)")
        print(f"kernel = np.ones(({self.morph_close_kernel}, {self.morph_close_kernel}), np.uint8)")
        print(f"closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)")

    def threshold_changed(self, threshold_value):
        self.threshold_value = threshold_value
        self.apply_operations()

    def gaussian_blur_ksize_changed(self, value):
        self.gaussian_blur_ksize = value
        self.apply_operations()

    def morph_open_kernel_changed(self, value):
        self.morph_open_kernel = value
        self.apply_operations()

    def morph_close_kernel_changed(self, value):
        self.morph_close_kernel = value
        self.apply_operations()

    def erosion_kernel_x_changed(self, value):
        self.erosion_kernel_x = value
        self.apply_operations()

    def erosion_kernel_y_changed(self, value):
        self.erosion_kernel_y = value
        self.apply_operations()

    def erosion_iterations_changed(self, value):
        self.erosion_iterations = value
        self.apply_operations()

    def dilation_kernel_x_changed(self, value):
        self.dilation_kernel_x = value
        self.apply_operations()

    def dilation_kernel_y_changed(self, value):
        self.dilation_kernel_y = value
        self.apply_operations()

    def dilation_iterations_changed(self, value):
        self.dilation_iterations = value
        self.apply_operations()

    def apply_operations(self):
        self.altered_image = apply_operations(
            self.original_image,
            self.gaussian_blur_ksize,
            self.threshold_value,
            self.erosion_kernel_x,
            self.erosion_kernel_y,
            self.erosion_iterations,
            self.dilation_kernel_x,
            self.dilation_kernel_y,
            self.dilation_iterations,
            self.morph_open_kernel,
            self.morph_close_kernel
        )
        self.update_altered_image_display(self.altered_image)
