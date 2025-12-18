from lr5 import CatImageProcessor
from lr5.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger("my_logger")

def main() -> None:
    cat_image_processor = CatImageProcessor()
    cat_image_processor.process_images_with_convolution(1)

main()

