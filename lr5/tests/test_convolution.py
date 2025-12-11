import unittest

import numpy as np

from lr5.core.entity.image_cat import ImageCatFactory, ImageCatGray, ImageCatRGB
from lr5.core.image_operations.convolution import Convolution


class TestConvolution(unittest.TestCase):
    def setUp(self):
        # Градиентное изображение для стабильных проверок
        self.gray = np.arange(0, 25, dtype=np.uint8).reshape(5, 5)
        self.rgb = np.dstack([self.gray, self.gray, self.gray])
        self.img_gray = ImageCatFactory.create_image_cat(
            index=1, filename="g", extension=".png", data=self.gray, url=None, breeds=[]
        )
        self.img_rgb = ImageCatFactory.create_image_cat(
            index=2, filename="r", extension=".png", data=self.rgb, url=None, breeds=[]
        )
        self.kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float)

    def test_convolution_gray(self):
        conv = Convolution(self.kernel_sharpen)
        out = conv.convolution(self.img_gray)
        self.assertIsInstance(out, ImageCatGray)
        self.assertEqual(out.data.shape, self.gray.shape)
        self.assertIn("_conv", out.filename)

    def test_convolution_rgb(self):
        conv = Convolution(self.kernel_sharpen)
        out = conv.convolution(self.img_rgb)
        self.assertIsInstance(out, ImageCatRGB)
        self.assertEqual(out.data.shape, self.rgb.shape)
        self.assertIn("_conv", out.filename)

    def test_convolution_cv2(self):
        conv = Convolution(self.kernel_sharpen)
        out = conv.convolution_cv2(self.img_rgb)
        self.assertIsInstance(out, ImageCatRGB)
        self.assertEqual(out.data.shape, self.rgb.shape)
        self.assertIn("_conv_cv2", out.filename)

    def test_invalid_kernel(self):
        with self.assertRaises(ValueError):
            Convolution(np.ones((3, 3, 3)))


if __name__ == "__main__":
    unittest.main()
