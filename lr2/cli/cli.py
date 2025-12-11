import asyncio
import logging

import click

from lr2.config import LOG_FILE_PATH
from lr2.core.service.cat_image_processor import CatImageProcessor

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
@click.option('-l', '--limit-images',
              required=True,
              type=int)
def detect_edges(limit_images: int):
    """Выделяет границы оператором Собеля"""
    cat_image_processor = CatImageProcessor()

    cat_image_processor.process_images_with_edges(limit_images)


@cli.command()
@click.option('-l', '--limit-images',
              required=True,
              type=int)
def convolution(limit_images: int):
    """Применяет свёртку к изображениям"""
    cat_image_processor = CatImageProcessor()

    cat_image_processor.process_images_with_convolution(limit_images)
    asyncio.run(cat_image_processor.process_images_with_convolution_async(limit_images))


@cli.command()
@click.option('-t', '--threshold',
              required=False,
              type=float,
              help="Порог для выделения углов")
@click.option('-l', '--limit-images',
              required=True,
              type=int)
def detect_corners(threshold: float, limit_images: int):
    """Выделяет углы на изображениях"""
    cat_image_processor = CatImageProcessor()

    cat_image_processor.process_images_with_corners(threshold, limit_images)


@cli.command()
@click.option('-g', '--gamma',
              required=False,
              type=float,
              help="Значение гамма для коррекции")
@click.option('-l', '--limit-images',
              required=True,
              type=int)
def gamma_correction(gamma: float, limit_images: int):
    """Применяет гамма-коррекцию к изображениям"""
    cat_image_processor = CatImageProcessor()

    cat_image_processor.process_images_with_gamma_correction(gamma, limit_images)


@cli.command()
@click.option('-l', '--limit-images',
              required=True,
              type=int)
def grayscale(limit_images: int):
    """Преобразует изображения в полутоновые"""
    cat_image_processor = CatImageProcessor()

    cat_image_processor.process_images_with_grayscale(limit_images)


if __name__ == "__main__":
    cli()
