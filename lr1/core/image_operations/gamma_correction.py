import cv2
import numpy as np

from lr1.core.entity.image import Image
from lr1.utils.performance_measurer import PerformanceMeasurer


class GammaCorrection:
    def __init__(self, gamma: float):
        if gamma <= 0:
            raise ValueError("Гамма должна быть положительным числом")
        self.gamma = float(gamma)
        self.inv_gamma = 1.0 / self.gamma

        # предвычисляем LUT для uint8 входа (0..255)
        lut = ((np.arange(256) / 255.0) ** (self.inv_gamma) * 255.0)
        self.lut_uint8 = np.clip(lut, 0, 255).astype(np.uint8)

    @PerformanceMeasurer.measure_time_decorator
    def gamma_correction(self, image: Image) -> Image:
        """Применяет гамма-коррекцию к изображению (grayscale или RGB)."""
        data = image.data

        # LUT применяется по каждому каналу
        out = self.lut_uint8[data]

        return Image(
            filename=image.filename + f"_gamma{self.gamma}",
            extension=image.extension,
            data=out
        )

    @PerformanceMeasurer.measure_time_decorator
    def gamma_correction_cv2(self, image: Image) -> Image:
        """Применяет гамма-коррекцию к изображению (grayscale или RGB)."""
        data = image.data

        out = cv2.LUT(data, self.lut_uint8)

        return Image(
            filename=image.filename + f"_gamma{self.gamma}_cv2",
            extension=image.extension,
            data=out
        )
