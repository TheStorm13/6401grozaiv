from abc import ABC

import numpy as np


class ImageCat(ABC):
    kernel = np.ones((3, 3)) / 100.0

    def __init__(self, index: int, filename: str, extension: str, data: np.ndarray, url: str, breeds: list[dict]):
        self.index = index
        self._filename = filename
        self.extension = extension
        self.data = data
        self.url = url
        self.breeds = breeds

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, filename):
        self._filename = filename

    def __add__(self, other):
        if not isinstance(other, ImageCat):
            raise TypeError("Операция сложения поддерживается только между объектами ImageCat")
        try:
            combined_data = self.data + other.data
        except ValueError as e:
            raise ValueError("Что ты творишь?") from e

        return ImageCatFactory.create_image_cat(
            index=self.index,
            filename=f"{self.filename}_plus_{other.filename}",
            extension=self.extension,
            data=combined_data,
            url=self.url,
            breeds=self.breeds + other.breeds
        )

    def __sub__(self, other):
        if not isinstance(other, ImageCat):
            raise TypeError("Операция вычитания поддерживается только между объектами ImageCat")
        try:
            subtracted_data = self.data - other.data
        except ValueError as e:
            raise ValueError("Что ты творишь?") from e

        return ImageCatFactory.create_image_cat(
            index=self.index,
            filename=f"{self.filename}_minus_{other.filename}",
            extension=self.extension,
            data=subtracted_data,
            url=self.url,
            breeds=self.breeds + other.breeds
        )

    def __str__(self) -> str:
        return f"ImageCat(filename={self.filename}, extension={self.extension}, shape={self.data.shape}, url={self.url})"

    @filename.setter
    def filename(self, value):
        self._filename = value


class ImageCatRGB(ImageCat):
    def apply_convolution(self, kernel: np.ndarray) -> np.ndarray:
        """Применяет свёртку к RGB-изображению."""
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        padded = np.pad(self.data, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant")
        out = np.zeros_like(self.data, dtype=float)
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                region = padded[i:i + kh, j:j + kw, :]
                out[i, j] = np.sum(region * kernel[:, :, None], axis=(0, 1))
        return np.clip(out, 0, 255).astype(np.uint8)


class ImageCatGray(ImageCat):
    def apply_convolution(self, kernel: np.ndarray) -> np.ndarray:
        """Применяет свёртку к изображению в градациях серого."""
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        padded = np.pad(self.data, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")

        out = np.zeros_like(self.data, dtype=float)

        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                region = padded[i:i + kh, j:j + kw]
                out[i, j] = np.sum(region * kernel)

        return np.clip(out, 0, 255).astype(np.uint8)


class ImageCatFactory:
    @staticmethod
    def create_image_cat(*args, data, **kwargs):
        if "index" not in kwargs:
            kwargs["index"] = 0
        if data.ndim == 2:
            return ImageCatGray(*args, data=data, **kwargs)
        elif data.ndim == 3:
            return ImageCatRGB(*args, data=data, **kwargs)
        else:
            raise ValueError(f"Неизвестная размерность данных: {data.ndim}")
