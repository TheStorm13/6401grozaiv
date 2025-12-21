import logging
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from lr1.config import IMAGE_EXTENSIONS
from lr1.core.entity.image import Image

logger = logging.getLogger(__name__)


class ImageStorage:
    def __init__(self, photo_dir: Path):
        """
        Инициализация хранилища изображений

        Args:
            photo_dir: Базовый каталог для изображений
        """

        self.photo_dir = photo_dir
        self.image_extensions = IMAGE_EXTENSIONS
        self.photo_dir.mkdir(parents=True, exist_ok=True)

    def _check_extension(self, path: Path) -> None:
        if path.suffix.lower() not in self.image_extensions:
            raise ValueError(f"Неподдерживаемый формат изображения: {path.suffix}")

    def _resolve_path(self, filename: str) -> Path:
        """Вернуть полный путь к файлу относительно photo_dir"""
        return self.photo_dir / filename

    def load_image(self, image_path: Path) -> Image:
        """
        Загрузить изображение и вернуть Image(filename, extension, data: np.ndarray).

        Raises:
            FileNotFoundError: если файл не найден.
            ValueError: если расширение не поддерживается или изображение не удалось прочитать.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Файл не найден: {image_path}")

        self._check_extension(image_path)

        try:
            with PILImage.open(image_path) as pil_img:
                pil_img.load()  # гарантировать чтение файла
                arr = np.asarray(pil_img)  # HxW или HxWxC
        except Exception as exc:
            logger.exception("Ошибка при загрузке изображения %s", image_path)
            raise ValueError(f"Не удалось загрузить изображение: {image_path}") from exc

        logger.info(
            "Изображение загружено: %s (shape=%s)", image_path.name, getattr(arr, "shape", None)
        )
        return Image(
            filename=image_path.stem,
            extension=image_path.suffix.lower(),
            data=arr,
        )

    def save_image(self, image: Image, output: Path) -> Path:
        """
        Сохранить numpy-изображение в файл в каталоге photo_dir.

        Args:
            image: экземпляр Image (filename, extension, data)
            output: путь до папки сохранения

        Returns:
            Path: полный путь к сохранённому файлу.

        Raises:
            ValueError: если расширение не поддерживается или данные неверного формата.
        """
        # todo: добавить сохранение по нужному пути
        dest = self._resolve_path(image.filename + image.extension)
        self._check_extension(dest)

        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            pil = PILImage.fromarray(image.data)
            pil.save(dest)
        except Exception as exc:
            logger.exception("Ошибка при сохранении изображения %s", dest)
            raise ValueError(f"Не удалось сохранить изображение: {dest}") from exc

        logger.info("Изображение сохранено: %s", dest)
        return dest
