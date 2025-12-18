# Экспортируем только главные классы для удобства
from .core.api.cat_api import CatAPI
from .core.entity.image_cat import ImageCat
from .core.service.cat_image_processor import CatImageProcessor

__all__ = ['CatAPI', 'CatImageProcessor', 'ImageCat']
