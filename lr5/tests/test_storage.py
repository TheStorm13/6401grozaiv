import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from lr5.core.entity.image_cat import ImageCatFactory
from lr5.core.storage.image_storage import ImageStorage


class TestImageStorage(unittest.TestCase):
    def setUp(self):
        """Создание временного каталога и тестового изображения."""
        self.tmpdir = Path(tempfile.mkdtemp())
        self.storage = ImageStorage(self.tmpdir)

        # Создаем тестовое RGB изображение 4x4
        self.data = (np.random.rand(4, 4, 3) * 255).astype(np.uint8)
        self.image = ImageCatFactory.create_image_cat(
            index=1, filename="test_img", extension=".jpg", data=self.data, url=None, breeds=[]
        )

    def test_save_image_sync(self):
        """Тест синхронного сохранения изображения."""
        out = self.storage.save_image(self.image, self.tmpdir / "sync")
        self.assertTrue(out.exists())
        self.assertEqual(out.suffix, ".jpg")

    def test_load_image(self):
        """Тест загрузки изображения с диска."""
        # Подготовим файл на диске и загрузим его
        img_path = self.tmpdir / "load" / "a.jpg"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        PILImage.fromarray(self.data).save(img_path)
        loaded = self.storage.load_image(img_path)
        self.assertEqual(loaded.filename, "a")
        self.assertEqual(loaded.extension, ".jpg")
        self.assertEqual(loaded.data.shape, self.data.shape)


if __name__ == "__main__":
    unittest.main()
