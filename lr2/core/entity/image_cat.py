from dataclasses import dataclass

import numpy as np


@dataclass
class ImageCat:
    filename: str
    extension: str
    data: np.ndarray
    url: str
    breeds: list[dict]

    def __add__(self, other):
        if not isinstance(other, ImageCat):
            raise TypeError("Операция сложения поддерживается только между объектами ImageCat")

        combined_data = self.data + other.data

        return ImageCat(
            filename=f"{self.filename}_plus_{other.filename}",
            extension=self.extension,
            data=combined_data,
            url=self.url,
            breeds=self.breeds + other.breeds
        )

    def __sub__(self, other):
        if not isinstance(other, ImageCat):
            raise TypeError("Операция вычитания поддерживается только между объектами ImageCat")

        subtracted_data = self.data - other.data

        return ImageCat(
            filename=f"{self.filename}_minus_{other.filename}",
            extension=self.extension,
            data=subtracted_data,
            url=self.url,
            breeds=self.breeds + other.breeds
        )

    def __str__(self)->str:
        return f"ImageCat(filename={self.filename}, extension={self.extension}, shape={self.data.shape}, url={self.url})"