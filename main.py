import sys
import cv2
import os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QWidget, QFileDialog)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QGridLayout, QLineEdit, QDialog
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtCore import Qt
plugin_path = QCoreApplication.libraryPaths()[0]


def count_planes(image):
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    b, _, _ = cv2.split(image)
    _, mask = cv2.threshold(b, 195, 255, cv2.THRESH_BINARY)
    result = cv2.bitwise_and(image, image, mask=mask)
    countours, _ = cv2.findContours(mask,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for countour in countours:
        if cv2.contourArea(countour) > 20:
            count += 1
            cv2.drawContours(result, [countour], -1, (0, 255, 0), 2)
    return count, result


class SetKernelDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Set Kernel')

        # Создание макета для диалогового окна
        layout = QGridLayout()

        # Метка и поле ввода для размера ядра
        self.kernel_size_label = QLabel('Kernel Size:', self)
        layout.addWidget(self.kernel_size_label, 0, 0)
        self.kernel_size_input = QLineEdit(self)
        layout.addWidget(self.kernel_size_input, 0, 1)

        # Метка и поле ввода для ядра
        self.kernel_elements_label = QLabel('Kernel Elements:', self)
        layout.addWidget(self.kernel_elements_label, 1, 0)
        self.kernel_elements_input = QLineEdit(self)
        layout.addWidget(self.kernel_elements_input, 1, 1)

        # Кнопка "OK" для закрытия диалогового окна
        self.ok_button = QPushButton('OK', self)
        self.ok_button.clicked.connect(self.accept)
        layout.addWidget(self.ok_button, 2, 1)

        # Установка макета для диалогового окна
        self.setLayout(layout)

    def get_kernel_data(self):
        # Получение данных о размере ядра и элементах ядра
        size = int(self.kernel_size_input.text())
        elements = list(map(float, self.kernel_elements_input.text().split()))

        if len(elements) > size * size:
            elements = elements[:size * size]
        elif len(elements) < size * size:
            elements += [1] * (size * size - len(elements))

        return size, elements


class PhotoTransformApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Photo Transformation App')
        self.setGeometry(100, 100, 1200, 800)

        self.image_label = QLabel(self)
        self.image_label.setScaledContents(False)

        self.load_button = QPushButton('Load Image', self)
        self.load_button.clicked.connect(self.load_image)
        self.load_button.setStyleSheet('background-color: blue; color: white;')

        self.erode_button = QPushButton('Erode', self)
        self.erode_button.clicked.connect(self.erode_image)
        self.erode_kernel_button = QPushButton('Установить ядро', self)
        self.erode_kernel_button.clicked.connect(self.set_erode_kernel)

        self.dilate_button = QPushButton('Dilate', self)
        self.dilate_button.clicked.connect(self.dilate_image)
        self.dilate_kernel_button = QPushButton('Установить ядро', self)
        self.dilate_kernel_button.clicked.connect(self.set_dilate_kernel)

        self.opening_button = QPushButton('Opening', self)
        self.opening_button.clicked.connect(self.opening_image)
        self.opening_kernel_button = QPushButton('Установить ядро', self)
        self.opening_kernel_button.clicked.connect(self.set_opening_kernel)

        self.closing_button = QPushButton('Closing', self)
        self.closing_button.clicked.connect(self.closing_image)
        self.closing_kernel_button = QPushButton('Установить ядро', self)
        self.closing_kernel_button.clicked.connect(self.set_closing_kernel)

        self.cusrtom_filter_button = QPushButton('Custom Filter', self)
        self.cusrtom_filter_button.clicked.connect(self.custom_filter_image)
        self.cusrtom_filter_kernel_button = QPushButton('Set Kernel', self)
        self.cusrtom_filter_kernel_button.clicked.connect(
            self.set_custom_filter_kernel)

        self.undo_button = QPushButton('Отменить', self)
        self.undo_button.clicked.connect(self.undo_action)

        self.reset_button = QPushButton('Вернуть оригинал', self)
        self.reset_button.clicked.connect(self.reset_image)

        self.count_airplanes = QPushButton('Посчитать самолеты', self)
        self.count_airplanes.clicked.connect(self.count_airplanes_skimage)

        self.erode_button.setStyleSheet(
            'QPushButton:hover { border: 2px solid green; color: green; }')
        self.dilate_button.setStyleSheet(
            'QPushButton:hover { border: 2px solid green; color: green; }')
        self.opening_button.setStyleSheet(
            'QPushButton:hover { border: 2px solid green; color: green; }')
        self.closing_button.setStyleSheet(
            'QPushButton:hover { border: 2px solid green; color: green; }')

        layout = QGridLayout()
        layout.addWidget(self.load_button, 0, 0, 1, 2)
        layout.addWidget(self.image_label, 1, 0, 1, 2)
        layout.addWidget(self.erode_button, 2, 0)
        layout.addWidget(self.erode_kernel_button, 2, 1)
        layout.addWidget(self.dilate_button, 3, 0)
        layout.addWidget(self.dilate_kernel_button, 3, 1)
        layout.addWidget(self.opening_button, 4, 0)
        layout.addWidget(self.opening_kernel_button, 4, 1)
        layout.addWidget(self.closing_button, 5, 0)
        layout.addWidget(self.closing_kernel_button, 5, 1)
        layout.addWidget(self.cusrtom_filter_button, 6, 0)
        layout.addWidget(self.cusrtom_filter_kernel_button, 6, 1)
        layout.addWidget(self.undo_button, 7, 0, 1, 2)
        layout.addWidget(self.reset_button, 8, 0, 1, 2)
        layout.addWidget(self.count_airplanes, 9, 0, 1, 2)

        self.erode_kernel = np.ones((5, 5), np.uint8)
        self.dilate_kernel = np.ones((5, 5), np.uint8)
        self.opening_kernel = np.ones((5, 5), np.uint8)
        self.closing_kernel = np.ones((5, 5), np.uint8)
        self.custom_filter_kernel = np.ones((3, 3), np.uint8)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.loaded_image = None
        self.history = []

    def load_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter('Images (*.png *.jpg *.jpeg *.bmp)')
        file_dialog.setViewMode(QFileDialog.Detail)
        file_dialog.setFileMode(QFileDialog.ExistingFile)

        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            self.image_file = file_path
            self.loaded_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            self.display_image(self.loaded_image)
            self.history = []

    def display_image(self, image):
        height, width, _ = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height,
                         bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        pixmap = pixmap.scaled(width,
                               height,
                               aspectRatioMode=Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

    def erode_image(self):
        if self.loaded_image is not None:
            self.history.append(self.loaded_image.copy())
            eroded_image = cv2.erode(
                self.loaded_image, self.erode_kernel, iterations=1)
            self.display_image(eroded_image)

    def dilate_image(self):
        if self.loaded_image is not None:
            self.history.append(self.loaded_image.copy())
            dilated_image = cv2.dilate(
                self.loaded_image, self.dilate_kernel, iterations=1)
            self.display_image(dilated_image)

    def opening_image(self):
        if self.loaded_image is not None:
            self.history.append(self.loaded_image.copy())
            opened_image = cv2.morphologyEx(
                self.loaded_image, cv2.MORPH_OPEN, self.opening_kernel)
            self.display_image(opened_image)

    def closing_image(self):
        if self.loaded_image is not None:
            self.history.append(self.loaded_image.copy())
            closed_image = cv2.morphologyEx(
                self.loaded_image, cv2.MORPH_CLOSE, self.closing_kernel)
            self.display_image(closed_image)

    def set_erode_kernel(self):
        dialog = SetKernelDialog()
        if dialog.exec_():
            size, elements = dialog.get_kernel_data()
            self.erode_kernel = np.array(elements).reshape(size, size)

    def set_dilate_kernel(self):
        dialog = SetKernelDialog()
        if dialog.exec_():
            size, elements = dialog.get_kernel_data()
            self.dilate_kernel = np.array(elements).reshape(size, size)

    def set_opening_kernel(self):
        dialog = SetKernelDialog()
        if dialog.exec_():
            size, elements = dialog.get_kernel_data()
            self.opening_kernel = np.array(elements).reshape(size, size)

    def set_closing_kernel(self):
        dialog = SetKernelDialog()
        if dialog.exec_():
            size, elements = dialog.get_kernel_data()
            self.closing_kernel = np.array(elements).reshape(size, size)

    def custom_filter_image(self):
        if self.loaded_image is not None:
            self.history.append(self.loaded_image.copy())
            kernel = self.custom_filter_kernel
            custom_filtered_image = cv2.filter2D(self.loaded_image, -1, kernel)
            self.display_image(custom_filtered_image)

    def set_custom_filter_kernel(self):
        dialog = SetKernelDialog()
        if dialog.exec_():
            size, elements = dialog.get_kernel_data()
            self.custom_filter_kernel = np.array(elements).reshape(size, size)

    def undo_action(self):
        if self.history:
            self.loaded_image = self.history.pop()
            self.display_image(self.loaded_image)

    def reset_image(self):
        if self.history:
            self.loaded_image = self.history[0].copy()
            self.history = []
            self.display_image(self.loaded_image)

    def count_airplanes_skimage(self):
        if self.loaded_image is not None:
            count, result = count_planes(self.loaded_image)
            self.display_image(result)
            self.statusBar().showMessage(f'Number of airplanes: {count}')
        else:
            self.statusBar().showMessage('No image is loaded')


if __name__ == '__main__':
    print(plugin_path)
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
    os.environ['QT_PLUGIN_PATH'] = plugin_path
    print("Initilizing app...")
    app = QApplication(sys.argv)
    print("App initialized")
    window = PhotoTransformApp()
    window.show()
    sys.exit(app.exec_())
