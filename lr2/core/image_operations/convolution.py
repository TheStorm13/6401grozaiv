import cv2
import numpy as np

from lr2.core.entity.image_cat import ImageCat
from lr2.utils.performance_measurer import PerformanceMeasurer


class Convolution:
    def __init__(self, kernel: np.ndarray):
        if kernel.ndim != 2:
            raise ValueError("Ядро должно быть двумерным")
        self.kernel = kernel.astype(float)

    @PerformanceMeasurer.measure_time_decorator
    def convolution(self, image: ImageCat) -> ImageCat:
        """Применяет свёртку к изображению (grayscale или RGB)."""
        data = image.data
        kh, kw = self.kernel.shape
        pad_h, pad_w = kh // 2, kw // 2

        # паддинг по краям нулями
        if data.ndim == 2:  # grayscale
            padded = np.pad(data, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
            out = np.zeros_like(data, dtype=float)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    region = padded[i:i + kh, j:j + kw]
                    out[i, j] = np.sum(region * self.kernel)

        elif data.ndim == 3:  # RGB
            padded = np.pad(data, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant")
            out = np.zeros_like(data, dtype=float)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    region = padded[i:i + kh, j:j + kw, :]
                    out[i, j] = np.sum(region * self.kernel[:, :, None], axis=(0, 1))
        else:
            raise ValueError("Формат изображения не поддерживается")

        out = np.clip(out, 0, 255).astype(np.uint8)

        return ImageCat(
            filename=image.filename + "_conv",
            extension=image.extension,
            data=out,
            url=image.url,
            breeds=image.breeds
        )

    @PerformanceMeasurer.measure_time_decorator
    def convolution_cv2(self, image: ImageCat) -> ImageCat:
        out = cv2.filter2D(image.data, -1, self.kernel)

        return ImageCat(
            filename=image.filename + "_conv_cv2",
            extension=image.extension,
            data=out,
            url=image.url,
            breeds=image.breeds
        )
