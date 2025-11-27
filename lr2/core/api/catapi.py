import io
import os
from typing import  Optional

import numpy as np
import requests
from PIL import Image as PILImage

from lr2.core.entity.image_cat import ImageCat


class CatAPI:
    # https://documenter.getpostman.com/view/5578104/RWgqUxxh
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "https://api.thecatapi.com/v1/images/search"
        self.api_key = api_key
        self.session = requests.Session()

        if self.api_key:
            self.session.headers.update({'x-api-key': self.api_key})

    def get_cats(self, limit: int = 1) -> list[dict]:
        """
        Получает список кошачьих изображений из API

        Args:
            limit: количество изображений (1-100)

        Returns:
            Список словарей с информацией об изображениях
        """
        params = {
            'size': 'low',
            'mime_types': 'jpg',
            'format': 'json',
            'has_breeds': True,
            'order': 'RANDOM',
            'page': 0,
            'limit': min(limit, 25)
        }

        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            images = response.json()

            return images[:limit]
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при запросе к API: {e}")
            return []

    def _get_image_data(self, image_url: str) -> Optional[np.ndarray]:
        """
        Получет данные изображения по URL и возвращает их в формате numpy.ndarray.

        Args:
            image_url: URL изображения

        Returns:
            numpy.ndarray или None в случае ошибки
        """
        try:
            response = self.session.get(image_url)
            response.raise_for_status()
            pil_image = PILImage.open(io.BytesIO(response.content))
            return np.asarray(pil_image)
        except Exception as e:
            print(f"Ошибка при скачивании изображения: {e}")
            return None

    def get_cat_images(self, limit: int = 1) -> list[ImageCat]:
        """
        Получает изображения с информацией о породе.

        Args:
            limit: количество изображений

        Returns:
            Список объектов ImageCat
        """
        images_data = self.get_cats(limit)
        downloaded_images = []

        for img_data in images_data:
            image_id = img_data['id']
            image_url = img_data['url']
            breeds = img_data.get('breeds', [])
            file_extension = os.path.splitext(image_url)[1]

            image_data = self._get_image_data(image_url)
            if image_data is None:
                continue


            # Создаем объект ImageCat
            image_cat = ImageCat(
                filename=image_id,
                extension=file_extension,
                data=image_data,
                url=image_url,
                breeds=breeds
            )

            downloaded_images.append(image_cat)

        return downloaded_images


