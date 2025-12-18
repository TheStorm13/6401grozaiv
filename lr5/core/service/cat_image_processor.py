import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from lr5.config import API_KEY
from lr5.config import PHOTO_DIR
from lr5.core.api.cat_api import CatAPI
from lr5.core.image_operations.convolution import Convolution
from lr5.core.image_operations.corner_detection import CornerDetection
from lr5.core.image_operations.edge_detection import EdgeDetection
from lr5.core.image_operations.gamma_correction import GammaCorrection
from lr5.core.image_operations.grayscale_converter import GrayscaleConverter
from lr5.core.storage.image_storage import ImageStorage
from lr5.utils.performance_measurer import PerformanceMeasurer

logger = logging.getLogger("my_logger")


class CatImageProcessor:
    def __init__(self, api_key=API_KEY):
        self.api = CatAPI(api_key)
        self.storage = ImageStorage(PHOTO_DIR)
        self.edge_detector = EdgeDetection()

        self.photo_dir = Path(PHOTO_DIR)
        self.originals_dir = self.photo_dir / "originals"
        self.manual_count_dir = self.photo_dir / "manual_count"
        self.cv2_dir = self.photo_dir / "cv2"

    def process_images_with_edges(self, limit: int = 5):
        """
        Главный метод для обработки изображений:
        1. Запрашивает изображения через API.
        2. Сохраняет оригиналы.
        3. Находит границы на изображениях.
        4. Сохраняет результаты.

        Args:
            limit: Количество изображений для обработки.
        """

        images = self.api.get_cat_images(limit=limit)
        if not images:
            logger.warning("Не удалось получить изображения от API (limit=%d).", limit)
            return

        for image in images:
            try:
                original_path = self.storage.save_image(image, self.originals_dir)
                logger.info("Оригинал сохранен: %s", original_path)

                edges_image = self.edge_detector.edge_detection(image)
                edges_path = self.storage.save_image(edges_image, self.manual_count_dir)
                logger.info("Границы (manual) сохранены: %s", edges_path)

                edges_image = self.edge_detector.edge_detection_cv2(image)
                edges_path = self.storage.save_image(edges_image, self.cv2_dir)
                logger.info("Границы (cv2) сохранены: %s", edges_path)
            except Exception as e:
                logger.exception("Ошибка при обработке изображения")

    @PerformanceMeasurer.measure_time_decorator
    def process_images_with_convolution(self, limit: int = 5):
        """
        Старая синхронная версия свёртки (оставлена для сравнения).
        Применяет свёртку к изображениям:
        1. Запрашивает изображения через API.
        2. Сохраняет оригиналы.
        3. Применяет свёртку с заданным ядром.
        4. Сохраняет результаты.

        Args:
            kernel: Ядро свёртки.
            limit: Количество изображений для обработки.
        """

        kernel = np.ones((3, 3)) / 100.0

        images = self.api.get_cat_images(limit=limit)
        if not images:
            logger.warning("Не удалось получить изображения от API (limit=%d).", limit)
            return

        convolution = Convolution(kernel)

        for image in images:
            try:
                original_path = self.storage.save_image(image, self.originals_dir)
                logger.info("Оригинал сохранен: %s", original_path)

                convolved_image = convolution.convolution(image)
                convolved_path = self.storage.save_image(convolved_image, self.manual_count_dir)
                logger.info("Свёртка (manual) сохранена: %s", convolved_path)
            except Exception as e:
                logger.exception("Ошибка при обработке изображения")

    @PerformanceMeasurer.measure_time_decorator
    async def process_images_with_convolution_async(self, limit: int = 5):
        """
        Новая версия:
        - асинхронное скачивание и сохранение оригиналов,
        - параллельная свёртка в процессах с сохранением порядка,
        - асинхронное сохранение результатов.
        """
        kernel = np.ones((3, 3)) / 100.0

        # 1. Асинхронно получаем изображения с фиксированными индексами
        images = await self.api.get_cat_images_async(limit=limit)
        if not images:
            logger.warning("Не удалось получить изображения от API (async, limit=%d).", limit)
            return

        # 2. Асинхронно сохраняем оригиналы
        save_tasks = [self.storage.save_image_async(img, self.originals_dir) for img in images]
        logger.info("Сохранение оригиналов (async) начато: count=%d", len(save_tasks))
        await asyncio.gather(*save_tasks)
        logger.info("Сохранение оригиналов (async) завершено")

        # 3. Параллельная свёртка: передаём минимальный набор данных
        args_list = []
        for idx, img in enumerate(images, start=1):
            args_list.append((idx, kernel, img.data))

        logger.info("Свёртка в процессах начата: count=%d", len(args_list))
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor() as pool:
            results = await asyncio.gather(*[
                loop.run_in_executor(pool, Convolution.run_convolution_task, args)
                for args in args_list
            ])
        logger.info("Свёртка в процессах завершена")

        # Создание объектов и асинхронное сохранение
        from lr5.core.entity.image_cat import ImageCatFactory
        save_conv_tasks = []
        for idx, convolved_data, suffix in results:
            img = images[idx - 1]
            out_img = ImageCatFactory.create_image_cat(
                index=img.index,
                filename=img.filename + suffix,
                extension=img.extension,
                data=convolved_data,
                url=img.url,
                breeds=img.breeds
            )
            save_conv_tasks.append(self.storage.save_image_async(out_img, self.manual_count_dir))

        logger.info("Сохранение результатов свёртки (async) начато: count=%d", len(save_conv_tasks))
        await asyncio.gather(*save_conv_tasks)
        logger.info("Сохранение результатов свёртки (async) завершено")

    def process_images_with_corners(self, threshold: float = 0.01, limit: int = 5):
        """
        Применяет детектор углов к изображениям:
        1. Запрашивает изображения через API.
        2. Сохраняет оригиналы.
        3. Находит углы на изображениях.
        4. Сохраняет результаты.

        Args:
            threshold: Порог для выделения углов.
            limit: Количество изображений для обработки.
        """
        images = self.api.get_cat_images(limit=limit)
        if not images:
            logger.warning("Не удалось получить изображения от API (limit=%d).", limit)
            return

        corner_detector = CornerDetection()

        for image in images:
            try:
                original_path = self.storage.save_image(image, self.originals_dir)
                logger.info("Оригинал сохранен: %s", original_path)

                corners_image = corner_detector.get_corners(image, threshold)
                corners_path = self.storage.save_image(corners_image, self.manual_count_dir)
                logger.info("Углы (manual) сохранены: %s", corners_path)

                corners_image_cv2 = corner_detector.corner_detection_cv2(image)
                corners_path_cv2 = self.storage.save_image(corners_image_cv2, self.cv2_dir)
                logger.info("Углы (cv2) сохранены: %s", corners_path_cv2)
            except Exception as e:
                logger.exception("Ошибка при обработке изображения")

    def process_images_with_gamma_correction(self, gamma: float = 10.0, limit: int = 5):
        """
        Применяет гамма-коррекцию к изображениям:
        1. Запрашивает изображения через API.
        2. Сохраняет оригиналы.
        3. Применяет гамма-коррекцию.
        4. Сохраняет результаты.

        Args:
            gamma: Значение гамма для коррекции.
            limit: Количество изображений для обработки.
        """
        images = self.api.get_cat_images(limit=limit)
        if not images:
            logger.warning("Не удалось получить изображения от API (limit=%d).", limit)
            return

        gamma_correction = GammaCorrection(gamma)

        for image in images:
            try:
                original_path = self.storage.save_image(image, self.originals_dir)
                logger.info("Оригинал сохранен: %s", original_path)

                gamma_corrected_image = gamma_correction.gamma_correction(image)
                gamma_corrected_path = self.storage.save_image(gamma_corrected_image, self.manual_count_dir)
                logger.info("Гамма-коррекция (manual) сохранена: %s", gamma_corrected_path)

                gamma_corrected_image_cv2 = gamma_correction.gamma_correction_cv2(image)
                gamma_corrected_path_cv2 = self.storage.save_image(gamma_corrected_image_cv2, self.cv2_dir)
                logger.info("Гамма-коррекция (cv2) сохранена: %s", gamma_corrected_path_cv2)
            except Exception as e:
                logger.exception("Ошибка при обработке изображения")

    def process_images_with_grayscale(self, limit: int = 5):
        """
        Преобразует изображения в полутоновые:
        1. Запрашивает изображения через API.
        2. Сохраняет оригиналы.
        3. Преобразует изображения в grayscale.
        4. Сохраняет результаты.

        Args:
            limit: Количество изображений для обработки.
        """
        images = self.api.get_cat_images(limit=limit)
        if not images:
            logger.warning("Не удалось получить изображения от API (limit=%d).", limit)
            return

        for image in images:
            try:
                original_path = self.storage.save_image(image, self.originals_dir)
                logger.info("Оригинал сохранен: %s", original_path)

                grayscale_image = GrayscaleConverter.to_grayscale(image)
                grayscale_path = self.storage.save_image(grayscale_image, self.manual_count_dir)
                logger.info("Grayscale (manual) сохранен: %s", grayscale_path)

                grayscale_image_cv2 = GrayscaleConverter.to_grayscale_cv2(image)
                grayscale_path_cv2 = self.storage.save_image(grayscale_image_cv2, self.cv2_dir)
                logger.info("Grayscale (cv2) сохранен: %s", grayscale_path_cv2)
            except Exception as e:
                logger.exception("Ошибка при обработке изображения")
