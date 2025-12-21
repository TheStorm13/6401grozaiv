import logging
import pathlib
from pathlib import Path

import click
import numpy as np
from lr1.config import LOG_FILE_PATH, PHOTO_DIR
from lr1.core.image_operations.convolution import Convolution
from lr1.core.image_operations.corner_detection import CornerDetection
from lr1.core.image_operations.edge_detection import EdgeDetection
from lr1.core.image_operations.gamma_correction import GammaCorrection
from lr1.core.image_operations.grayscale_converter import GrayscaleConverter
from lr1.core.storage.image_storage import ImageStorage

logging.basicConfig(
    level=logging.INFO,
    force=True,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='w', encoding="utf-8"),
        logging.StreamHandler()
    ]
)


@click.group()
@click.version_option("1.0.0")
def cli():
    """Manage your project with ease."""


@cli.command()
@click.argument('image_path',
                required=True,
                type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path, resolve_path=True))
@click.option('-o', '--output',
              type=click.Path(dir_okay=False, path_type=pathlib.Path),
              help='Путь для сохранения результата')
def convolution(image_path: Path, output: Path = None):
    """Применяет свертку к изображению"""
    storage = ImageStorage(PHOTO_DIR)
    image = storage.load_image(image_path)

    # ядро 3x3
    kernel = np.ones((3, 3)) / 100.0
    conv = Convolution(kernel)

    result = conv.convolution(image)
    storage.save_image(result, output)

    result = conv.convolution_cv2(image)
    storage.save_image(result, output)


@cli.command()
@click.argument('image_path',
                required=True,
                type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path, resolve_path=True))
@click.option('-o', '--output',
              type=click.Path(dir_okay=False, path_type=pathlib.Path),
              help='Путь для сохранения результата')
def grayscale(image_path: Path, output: Path = None):
    """Конвертирует в полутоновое изображение"""
    storage = ImageStorage(PHOTO_DIR)
    image = storage.load_image(image_path)

    result = GrayscaleConverter.to_grayscale(image)
    storage.save_image(result, output)

    result = GrayscaleConverter.to_grayscale_cv2(image)
    storage.save_image(result, output)


@cli.command()
@click.argument('image_path',
                required=True,
                type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path, resolve_path=True))
@click.option('-o', '--output',
              type=click.Path(dir_okay=False, path_type=pathlib.Path),
              help='Путь для сохранения результата')
def gamma_correction(image_path: Path, output: Path = None):
    """Применяет гамма-коррекцию"""
    storage = ImageStorage(PHOTO_DIR)
    image = storage.load_image(image_path)

    gamma_correction = GammaCorrection(10)

    result = gamma_correction.gamma_correction(image)
    storage.save_image(result, output)

    result = gamma_correction.gamma_correction_cv2(image)
    storage.save_image(result, output)


@cli.command()
@click.argument('image_path',
                required=True,
                type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path, resolve_path=True))
@click.option('-o', '--output',
              type=click.Path(dir_okay=False, path_type=pathlib.Path),
              help='Путь для сохранения результата')
def detect_edges(image_path: Path, output: Path = None):
    """Выделяет границы оператором Собеля"""
    storage = ImageStorage(PHOTO_DIR)
    image = storage.load_image(image_path)

    edge_detection = EdgeDetection()

    result = edge_detection.edge_detection(image)
    storage.save_image(result, output)

    result = edge_detection.edge_detection_cv2(image)
    storage.save_image(result, output)


@cli.command()
@click.argument('image_path',
                required=True,
                type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path, resolve_path=True))
@click.option('-o', '--output',
              type=click.Path(dir_okay=False, path_type=pathlib.Path),
              help='Путь для сохранения результата')
def detect_corners(image_path: Path, output: Path = None):
    """Обнаруживает углы детектором Харриса"""
    storage = ImageStorage(PHOTO_DIR)
    image = storage.load_image(image_path)

    corner_detection = CornerDetection()

    result = corner_detection.get_corners(image)
    storage.save_image(result, output)

    result = corner_detection.corner_detection_cv2(image)
    storage.save_image(result, output)


@cli.command()
@click.argument('image_path',
                required=True,
                type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path, resolve_path=True))
@click.option('-o', '--output',
              type=click.Path(dir_okay=False, path_type=pathlib.Path),
              help='Путь для сохранения результата')
def detect_circles(image_path: Path, output: Path = None):
    """Находит круги преобразованием Хафа"""
    pass

if __name__ == "__main__":
    cli()
