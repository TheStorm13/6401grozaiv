import asyncio
import io
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from PIL import Image as PILImage

from lr5.core.api.cat_api import CatAPI


class TestCatAPI(unittest.TestCase):
    def setUp(self):
        self.api = CatAPI(api_key=None)

    @patch("requests.Session.get")
    def test_get_cat_images_sync(self, mock_get):
        # Метаответ для get_cats
        meta = [{"id": "abc", "url": "http://x/img.jpg", "breeds": [{"name": "a"}]}]
        # Первый вызов: метаданные
        resp_meta = MagicMock()
        resp_meta.json.return_value = meta
        resp_meta.raise_for_status.return_value = None

        # Второй вызов: загрузка изображения
        arr = (np.random.rand(4, 4, 3) * 255).astype(np.uint8)
        buf = io.BytesIO()
        PILImage.fromarray(arr).save(buf, format="JPEG")
        resp_img = MagicMock()
        resp_img.content = buf.getvalue()
        resp_img.raise_for_status.return_value = None

        mock_get.side_effect = [resp_meta, resp_img]

        images = self.api.get_cat_images(limit=1)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].filename, "abc")
        self.assertEqual(images[0].extension, ".jpg")

    @patch("aiohttp.ClientSession.get")
    @patch.object(CatAPI, "get_cats")
    def test_get_cat_images_async(self, mock_get_cats, mock_session_get):
        # Мета
        mock_get_cats.return_value = [{"id": "abc", "url": "http://x/img.jpg", "breeds": []}]

        # Ответ для ClientSession.get
        class FakeResp:
            def __init__(self, data):
                self._data = data

            async def read(self):
                return self._data

            def raise_for_status(self):
                return None

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        # Подготовим байты JPEG
        arr = (np.random.rand(4, 4, 3) * 255).astype(np.uint8)
        buf = io.BytesIO()
        PILImage.fromarray(arr).save(buf, format="JPEG")
        mock_session_get.return_value = FakeResp(buf.getvalue())

        async def run():
            out = await self.api.get_cat_images_async(limit=1)
            return out

        images = asyncio.run(run())
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].filename, "abc")
        self.assertEqual(images[0].extension, ".jpg")


if __name__ == "__main__":
    unittest.main()
