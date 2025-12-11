import logging
import os

import cv2
import numpy as np

from lr2.core.entity.image_cat import ImageCatFactory
from lr2.utils.performance_measurer import PerformanceMeasurer

logger = logging.getLogger(__name__)


class Convolution:
    def __init__(self, kernel: np.ndarray):
        if kernel.ndim != 2:
            raise ValueError("Ядро должно быть двумерным")
        self.kernel = kernel.astype(float)

    @PerformanceMeasurer.measure_time_decorator
    def convolution(self, image):
        """Применяет свёртку к изображению (grayscale или RGB)."""
        if not hasattr(image, 'apply_convolution'):
            raise ValueError("Класс изображения не поддерживает свёртку")

        out = image.apply_convolution(self.kernel)

        return ImageCatFactory.create_image_cat(
            index=image.index,
            filename=image.filename + "_conv",
            extension=image.extension,
            data=out,
            url=image.url,
            breeds=image.breeds
        )

    @PerformanceMeasurer.measure_time_decorator
    def convolution_cv2(self, image):
        out = cv2.filter2D(image.data, -1, self.kernel)

        return ImageCatFactory.create_image_cat(
            index=image.index,
            filename=image.filename + "_conv_cv2",
            extension=image.extension,
            data=out,
            url=image.url,
            breeds=image.breeds
        )

    @staticmethod
    def run_convolution_task(args: tuple):
        """
        Рабочая функция для ProcessPoolExecutor, отдельно от методов класса.
        """
        idx, kernel, data = args
        logger.info("Свёртка (Process) начата: idx=%d, pid=%d", idx, os.getpid())
        out = cv2.filter2D(data, -1, kernel.astype(float))
        logger.info("Свёртка (Process) завершена: idx=%d, pid=%d", idx, os.getpid())
        return idx, out, "_conv"
