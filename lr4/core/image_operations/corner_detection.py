import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from lr4.core.entity.image_cat import ImageCat
from lr4.core.entity.image_cat import ImageCatFactory
from lr4.core.image_operations.convolution import Convolution
from lr4.core.image_operations.grayscale_converter import GrayscaleConverter
from lr4.utils.performance_measurer import PerformanceMeasurer


# todo: проверить правильность работы
class CornerDetection:
    """Выделение углов изображения с помощью детектора Харриса."""

    # Ядра Собеля для градиентов
    SOBEL_X = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]], dtype=float)

    SOBEL_Y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]], dtype=float)

    def __init__(self, k: float = 0.04, sigma: float = 1.0, nms_radius: int = 0.5):
        """
        k: параметр Харриса (обычно 0.04-0.06)
        sigma: стандартное отклонение для гауссова сглаживания
        nms_radius: радиус окрестности для подавления немаксимумов
        """
        self.k = k
        self.sigma = sigma
        self.conv_x = Convolution(CornerDetection.SOBEL_X)
        self.conv_y = Convolution(CornerDetection.SOBEL_Y)
        self.nms_radius = max(1, int(nms_radius))

    def corner_detection(self, image) -> np.ndarray:
        """Возвращает карту отклика углов R"""

        # Проверка, что изображение grayscale
        gray = GrayscaleConverter.to_grayscale(image)

        # Градиенты
        Ix = self.conv_x.convolution(ImageCatFactory.create_image_cat("", "", gray)).data.astype(float)
        Iy = self.conv_y.convolution(ImageCatFactory.create_image_cat("", "", gray)).data.astype(float)

        # Сглаживаем квадраты градиентов
        Sxx = self._gaussian_blur(Ix * Ix, self.sigma)
        Syy = self._gaussian_blur(Iy * Iy, self.sigma)
        Sxy = self._gaussian_blur(Ix * Iy, self.sigma)

        # Отклик Харриса
        det_M = Sxx * Syy - Sxy ** 2
        trace_M = Sxx + Syy
        R = det_M - self.k * (trace_M ** 2)

        return R

    @PerformanceMeasurer.measure_time_decorator
    def get_corners(self, image, threshold: float = 0.01):
        """
        Возвращает координаты углов (row, col), прошедших порог и NMS.
        threshold — доля от максимального положительного R (0..1).
        """
        R = self.corner_detection(image)

        R_max = np.max(R)
        if R_max <= 0 or not np.isfinite(R_max):
            return image  # углов нет

        corner_threshold = threshold * R_max
        mask = R > corner_threshold

        local_max = maximum_filter(R, size=2 * self.nms_radius + 1)
        peaks = (R == local_max) & mask
        corners = np.argwhere(peaks)

        # Делаем копию изображения в цвете
        result = GrayscaleConverter.to_rgb(image).data.astype(np.uint8)

        # Рисуем углы красным
        for y, x in corners:  # (row, col)
            cv2.circle(result, (int(x), int(y)), 2, (0, 0, 255), -1)

        return ImageCat(
            index=image.index,
            filename=image.filename + "_corn",
            extension=image.extension,
            data=result,
            url=image.url,
            breeds=image.breeds
        )

    def _gaussian_blur(self, data: np.ndarray, sigma: float) -> np.ndarray:
        """Простая реализация гауссова сглаживания через ядро."""
        return gaussian_filter(data, sigma=sigma)

    @PerformanceMeasurer.measure_time_decorator
    def corner_detection_cv2(self, image):

        gray = GrayscaleConverter.to_grayscale_cv2(image).data
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        result = image.data.copy()
        result[dst > 0.01 * dst.max()] = [255, 0, 0]

        return ImageCatFactory.create_image_cat(
            index=image.index,
            filename=image.filename + "_corn_cv2",
            extension=image.extension,
            data=result,
            url=image.url,
            breeds=image.breeds
        )
