import cv2
import numpy as np

from lr5.core.entity.image_cat import ImageCatFactory
from lr5.core.entity.image_cat import ImageCatGray
from lr5.core.entity.image_cat import ImageCatRGB
from lr5.utils.performance_measurer import PerformanceMeasurer


class GrayscaleConverter:
    """
    Преобразует цветное изображение RGB в полутоновое (grayscale).
    """

    @PerformanceMeasurer.measure_time_decorator
    @staticmethod
    def to_grayscale(image):
        if isinstance(image, ImageCatGray):
            gray_data = image.data.copy()
        elif isinstance(image, ImageCatRGB):
            gray_data = (0.299 * image.data[:, :, 0] +
                         0.587 * image.data[:, :, 1] +
                         0.114 * image.data[:, :, 2])
            gray_data = np.clip(gray_data, 0, 255).astype(np.uint8)
        else:
            raise ValueError("Изображение должно быть RGB или grayscale")

        return ImageCatFactory.create_image_cat(
            index=image.index,
            filename=image.filename + "_gray",
            extension=image.extension,
            data=gray_data,
            url=image.url,
            breeds=image.breeds
        )

    @PerformanceMeasurer.measure_time_decorator
    @staticmethod
    def to_grayscale_cv2(image):
        if isinstance(image, ImageCatGray):
            gray_data = image.data.copy()
        elif isinstance(image, ImageCatRGB):
            gray_data = cv2.cvtColor(image.data, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError("Изображение должно быть RGB или grayscale")

        return ImageCatFactory.create_image_cat(
            index=image.index,
            filename=image.filename + "_gray_cv2",
            extension=image.extension,
            data=gray_data,
            url=image.url,
            breeds=image.breeds
        )

    @PerformanceMeasurer.measure_time_decorator
    @staticmethod
    def to_rgb(image):
        if isinstance(image, ImageCatGray):
            rgb_data = cv2.cvtColor(image.data.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        elif isinstance(image, ImageCatRGB):
            rgb_data = image.data.copy()

        else:
            raise ValueError("Изображение должно быть RGB или grayscale")

        return ImageCatFactory.create_image_cat(
            index=image.index,
            filename=image.filename + "_rgb",
            extension=image.extension,
            data=rgb_data,
            url=image.url,
            breeds=image.breeds
        )
