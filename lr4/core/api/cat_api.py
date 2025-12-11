import asyncio
import io
import logging
import os
from typing import Optional

import aiohttp
import numpy as np
import requests
from PIL import Image as PILImage

from lr4.core.entity.image_cat import ImageCatFactory, ImageCat
from lr4.utils.performance_measurer import PerformanceMeasurer

logger = logging.getLogger(__name__)


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
            logger.info("Получены метаданные изображений: count=%d (limit=%d)", len(images[:limit]), limit)
            return images[:limit]
        except requests.exceptions.RequestException as e:
            logger.exception("Ошибка при запросе к API")
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
            logger.exception("Ошибка загрузки изображения по URL: %s", image_url)
            return None

    @PerformanceMeasurer.measure_time_decorator
    def get_cat_images(self, limit: int = 1) -> list:
        """
        Синхронная версия: получает изображения с информацией о породе.
        """
        images_data = self.get_cats(limit)
        downloaded_images = []

        for i, img_data in enumerate(images_data, start=1):
            image_id = img_data['id']
            image_url = img_data['url']
            breeds = img_data.get('breeds', [])
            file_extension = os.path.splitext(image_url)[1]

            image_data = self._get_image_data(image_url)
            if image_data is None:
                logger.warning("Пропуск изображения (нет данных): id=%s, url=%s", image_id, image_url)
                continue

            image_cat = ImageCatFactory.create_image_cat(
                index=i,
                filename=image_id,
                extension=file_extension,
                data=image_data,
                url=image_url,
                breeds=breeds
            )
            downloaded_images.append(image_cat)

        logger.info("Собрано изображений: %d (limit=%d)", len(downloaded_images), limit)
        return downloaded_images

    @staticmethod
    async def fetch_image(session: aiohttp.ClientSession, item: dict) -> tuple[int, bytes]:
        idx = item["index"]
        url = item["url"]
        logger.info("Загрузка изображения (async) начата: idx=%d", idx)
        async with session.get(url) as resp:
            resp.raise_for_status()
            data = await resp.read()
        logger.info("Загрузка изображения (async) завершена: idx=%d", idx)
        return idx, data

    @staticmethod
    def to_numpy(data_bytes: bytes) -> np.ndarray:
        pil_image = PILImage.open(io.BytesIO(data_bytes))
        return np.asarray(pil_image)

    @PerformanceMeasurer.measure_time_decorator
    async def get_cat_images_async(self, limit: int = 1) -> list[ImageCat]:
        """
        Асинхронная версия: получает список изображений и формирует объекты ImageCat.
        Индексы закрепляются при получении списка URL.

        Args:
            limit: количество изображений
            save_dir: папка для сохранения исходных файлов; если None — не сохранять

        Returns:
            Список объектов ImageCat в исходном порядке (по закрепленным индексам)
        """
        images_meta = self.get_cats(limit)
        indexed = [
            {
                "index": i,
                "id": meta["id"],
                "url": meta["url"],
                "breeds": meta.get("breeds", []),
                "ext": os.path.splitext(meta["url"])[1] or ".jpg",
            }
            for i, meta in enumerate(images_meta, start=1)
        ]

        async with aiohttp.ClientSession(headers={'x-api-key': self.api_key} if self.api_key else None) as session:
            logger.info("Начало асинхронной загрузки: count=%d", len(indexed))
            tasks = [self.fetch_image(session, item) for item in indexed]
            results: list[tuple[int, bytes]] = await asyncio.gather(*tasks, return_exceptions=False)
            logger.info("Асинхронная загрузка завершена")

            image_objs: list[Optional[ImageCat]] = [None] * len(indexed)
            for (idx, data), item in zip(results, indexed):
                try:
                    arr = self.to_numpy(data)
                    image_cat = ImageCatFactory.create_image_cat(
                        filename=item["id"],
                        extension=item["ext"],
                        data=arr,
                        url=item["url"],
                        breeds=item["breeds"]
                    )
                    image_objs[idx - 1] = image_cat
                    logger.debug("Преобразование в ImageCat: idx=%d, id=%s", idx, item["id"])
                except Exception as e:
                    logger.exception("Ошибка обработки изображения idx=%d id=%s", idx, item["id"])

        return [img for img in image_objs if img is not None]
