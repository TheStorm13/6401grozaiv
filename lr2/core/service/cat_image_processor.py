from pathlib import Path

from lr2.config import PHOTO_DIR
from lr2.core.api.catapi import CatAPI
from lr2.core.image_operations.edge_detection import EdgeDetection
from lr2.core.storage.image_storage import ImageStorage
from lr2.config  import API_KEY


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
            print("Не удалось получить изображения.")
            return

        for image in images:
            try:
                original_path = self.storage.save_image(image, self.originals_dir)
                print(f"Оригинал сохранен: {original_path}")

                edges_image = self.edge_detector.edge_detection(image)

                edges_path = self.storage.save_image(edges_image, self.manual_count_dir)
                print(f"Результат с самописным подсчетом границ сохранен: {edges_path}")

                edges_image = self.edge_detector.edge_detection_cv2(image)

                edges_path = self.storage.save_image(edges_image, self.cv2_dir)
                print(f"Результат с подсчетом границ cv2 сохранен: {edges_path}")

                print("-" * 40)

            except Exception as e:
                print(f"Ошибка при обработке изображения {image.filename}: {e}")


# Пример использования
if __name__ == "__main__":
    cat_image_processor = CatImageProcessor()

    cat_image_processor.process_images_with_edges(4)
