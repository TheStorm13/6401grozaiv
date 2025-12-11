import cv2
import numpy as np

from lr5.core.entity.image_cat import ImageCatFactory
from lr5.core.image_operations.convolution import Convolution
from lr5.core.image_operations.grayscale_converter import GrayscaleConverter
from lr5.utils.performance_measurer import PerformanceMeasurer


class EdgeDetection:
    """Выделение границ изображения с помощью оператора Собеля."""

    # Классические ядра Собеля
    SOBEL_X = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=float)

    SOBEL_Y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=float)

    def __init__(self):
        self.conv_x = Convolution(EdgeDetection.SOBEL_X)
        self.conv_y = Convolution(EdgeDetection.SOBEL_Y)

    @PerformanceMeasurer.measure_time_decorator
    def edge_detection(self, image):
        """Применяет оператор Собеля к изображению."""

        gray_image = GrayscaleConverter.to_grayscale(image)

        data = gray_image.data

        temp_im = ImageCatFactory.create_image_cat(filename=image.filename,
                                                   extension=image.extension,
                                                   data=data,
                                                   url=image.url,
                                                   breeds=image.breeds)

        gx_image = self.conv_x.convolution(temp_im)
        gy_image = self.conv_y.convolution(temp_im)

        gx = gx_image.data.astype(float)
        gy = gy_image.data.astype(float)

        magnitude = np.hypot(gx, gy)

        max_val = magnitude.max() if magnitude.size > 0 else 0.0
        if max_val == 0:
            normalized = np.zeros_like(magnitude, dtype=np.uint8)
        else:
            normalized = (magnitude / max_val) * 255.0
            normalized = np.clip(normalized, 0, 255).astype(np.uint8)

        return ImageCatFactory.create_image_cat(
            index=image.index,
            filename=image.filename + "_edge",
            extension=image.extension,
            data=normalized,
            url=image.url,
            breeds=image.breeds
        )

    @PerformanceMeasurer.measure_time_decorator
    def edge_detection_cv2(self, image):
        """Применяет оператор Собеля к изображению."""

        gray = GrayscaleConverter.to_grayscale_cv2(image).data
        edges = cv2.Canny(gray, 100, 200)
        out = edges

        return ImageCatFactory.create_image_cat(
            index=image.index,
            filename=image.filename + "_edge_cv2",
            extension=image.extension,
            data=out,
            url=image.url,
            breeds=image.breeds
        )
