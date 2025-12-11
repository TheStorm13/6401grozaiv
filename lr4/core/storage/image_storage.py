import logging
from io import BytesIO
from pathlib import Path

import aiofiles
import numpy as np
from PIL import Image as PILImage

from lr4.config import IMAGE_EXTENSIONS
from lr4.core.entity.image_cat import ImageCatFactory

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
            logger.error(f"Неподдерживаемый формат изображения: {path.suffix}")
            raise ValueError(f"Неподдерживаемый формат изображения: {path.suffix}")

    def load_image(self, image_path: Path):
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

        logger.info("Изображение загружено: %s", image_path.name)
        return ImageCatFactory.create_image_cat(
            index=0,
            filename=image_path.stem,
            extension=image_path.suffix.lower(),
            data=arr,
            url=None,
            breeds=[]
        )

    def save_image(self, image, output: Path = None) -> Path:
        """
        Сохранить numpy-изображение в файл в указанном каталоге или в photo_dir по умолчанию.

        Args:
            image: экземпляр Image (filename, extension, data)
            output: путь до папки сохранения (если не указан, используется photo_dir)

        Returns:
            Path: полный путь к сохранённому файлу.

        Raises:
            ValueError: если расширение не поддерживается или данные неверного формата.
        """

        save_dir = output if output else self.photo_dir
        dest = save_dir / (image.filename + image.extension)

        try:
            self._check_extension(dest)
        except ValueError as exc:
            raise exc

        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            pil = PILImage.fromarray(image.data)
            pil.save(dest)
        except Exception as exc:
            logger.exception("Ошибка при сохранении изображения %s", dest)
            raise ValueError(f"Не удалось сохранить изображение: {dest}") from exc

        logger.info("Изображение сохранено: %s", dest)
        return dest

    async def save_image_async(self, image, output: Path = None) -> Path:
        """
        Асинхронно сохранить numpy-изображение в файл.

        Args:
            image: экземпляр Image (filename, extension, data)
            output: путь до папки сохранения

        Returns:
            Path: полный путь к сохранённому файлу.
        """

        save_dir = output if output else self.photo_dir
        dest = save_dir / (image.filename + image.extension)

        self._check_extension(dest)
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            buf = BytesIO()
            ext = (image.extension or "").lower().lstrip(".")
            format_map = {
                "jpg": "JPEG",
                "jpeg": "JPEG",
                "png": "PNG",
            }
            pil_format = format_map.get(ext, None)
            logger.debug("Начало асинхронного сохранения: dest=%s, format=%s", dest, pil_format)
            PILImage.fromarray(image.data).save(buf, format=pil_format)
            data_bytes = buf.getvalue()
            async with aiofiles.open(dest, "wb") as f:
                await f.write(data_bytes)
            logger.info("Асинхронно сохранено изображение: %s", dest)
        except Exception as exc:
            logger.exception("Ошибка при асинхронном сохранении изображения %s", dest)
            raise ValueError(f"Не удалось сохранить изображение: {dest}") from exc

        return dest
