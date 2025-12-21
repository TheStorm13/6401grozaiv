import cv2
import numpy as np

from lr1.core.entity.image import Image
from lr1.core.image_operations.convolution import Convolution
from lr1.core.image_operations.grayscale_converter import GrayscaleConverter
from lr1.utils.performance_measurer import PerformanceMeasurer


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
    def edge_detection(self, image: Image) -> Image:
        """Применяет оператор Собеля к изображению."""

        data = image.data

        if data.ndim == 3:
            gray_img = GrayscaleConverter.to_grayscale_cv2(image)
            data = gray_img.data

        # Создаем временный Image
        temp_im = Image(filename=image.filename, extension=image.extension, data=data)

        gx_image = self.conv_x.convolution(temp_im)
        gy_image = self.conv_y.convolution(temp_im)

        gx = gx_image.data.astype(float)
        gy = gy_image.data.astype(float)

        magnitude = np.hypot(gx, gy)  # float

        max_val = magnitude.max() if magnitude.size > 0 else 0.0
        if max_val == 0:
            normalized = np.zeros_like(magnitude, dtype=np.uint8)
        else:
            normalized = (magnitude / max_val) * 255.0
            normalized = np.clip(normalized, 0, 255).astype(np.uint8)

        return Image(
            filename=image.filename + "_edge",
            extension=image.extension,
            data=normalized
        )

    @PerformanceMeasurer.measure_time_decorator
    def edge_detection_cv2(self, image: Image) -> Image:
        """Применяет оператор Собеля к изображению."""

        gray = GrayscaleConverter.to_grayscale_cv2(image).data
        edges = cv2.Canny(gray, 100, 200)
        out = edges

        return Image(
            filename=image.filename + "_edge_cv2",
            extension=image.extension,
            data=out
        )
