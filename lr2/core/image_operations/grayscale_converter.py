import cv2
import numpy as np

from lr2.core.entity.image_cat import ImageCat
from lr2.utils.performance_measurer import PerformanceMeasurer


class GrayscaleConverter:
    """
    Преобразует цветное изображение RGB в полутоновое (grayscale).
    """

    @PerformanceMeasurer.measure_time_decorator
    @staticmethod
    def to_grayscale(image: ImageCat) -> ImageCat:
        data = image.data

        if data.ndim == 2:  # уже grayscale
            gray_data = data.copy()

        elif data.ndim == 3 and data.shape[2] == 3:  # RGB
            # Преобразуем к полутоновому с использованием стандартных коэффициентов
            gray_data = (0.299 * data[:, :, 0] +
                         0.587 * data[:, :, 1] +
                         0.114 * data[:, :, 2])
            gray_data = np.clip(gray_data, 0, 255).astype(np.uint8)
        else:
            raise ValueError("Изображение должно быть RGB или grayscale")

        return ImageCat(
            filename=image.filename + "_gray",
            extension=image.extension,
            data=gray_data,
            url=image.url,
            breeds=image.breeds
        )

    @PerformanceMeasurer.measure_time_decorator
    @staticmethod
    def to_grayscale_cv2(image: ImageCat) -> ImageCat:
        data = image.data

        gray_data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)

        return ImageCat(
            filename=image.filename + "_gray_cv2",
            extension=image.extension,
            data=gray_data,
            url=image.url,
            breeds=image.breeds
        )
