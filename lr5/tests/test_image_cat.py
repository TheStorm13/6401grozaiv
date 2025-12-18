import unittest

import numpy as np

from lr5.core.entity.image_cat import ImageCatFactory, ImageCatRGB, ImageCatGray


class TestImageCat(unittest.TestCase):
    def setUp(self):
        """Подготовка тестовых изображений RGB и Gray."""
        # Простейшие данные для тестов
        self.rgb = np.array([
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ], dtype=np.uint8)
        self.gray = np.array([
            [10, 40],
            [70, 100],
        ], dtype=np.uint8)

        self.img_rgb = ImageCatFactory.create_image_cat(
            index=1, filename="rgb", extension=".jpg", data=self.rgb, url="", breeds=[]
        )
        self.img_gray = ImageCatFactory.create_image_cat(
            index=2, filename="gray", extension=".jpg", data=self.gray, url="", breeds=[]
        )

    def test_factory_rgb_bw_conversion_by_shape(self):
        """Проверка фабрики: RGB и Gray по форме массива."""
        # RGB остается RGB, а 2D массив -> Gray
        self.assertIsInstance(self.img_rgb, ImageCatRGB)
        self.assertIsInstance(self.img_gray, ImageCatGray)

    def test_convolution_rgb_and_gray(self):
        """Тест применения свёртки к RGB и Gray изображениям."""
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=float)

        out_rgb = self.img_rgb.apply_convolution(kernel)
        self.assertEqual(out_rgb.shape, self.rgb.shape)

        out_gray = self.img_gray.apply_convolution(kernel)
        self.assertEqual(out_gray.shape, self.gray.shape)

    def test_add_images_same_type(self):
        """Тест сложения изображений одного типа."""
        # Сложение RGB+RGB -> RGB
        res = self.img_rgb + self.img_rgb
        self.assertIsInstance(res, ImageCatRGB)
        self.assertEqual(res.data.shape, self.rgb.shape)
        self.assertIn("_plus_", res.filename)

    def test_add_images_mismatch_raises(self):
        """Тест ошибки при сложении с несовместимым типом."""
        # Сложение с не-ImageCat
        with self.assertRaises(TypeError):
            _ = self.img_rgb + 123


if __name__ == "__main__":
    unittest.main()
