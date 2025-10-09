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


if __name__ == "__main__":
    cli()
