import logging
from logging import Handler, Formatter
from logging.handlers import RotatingFileHandler
from pathlib import Path

from lr5.config import LOG_FILE_PATH


def setup_logging() -> None:
    """
    Глобальная настройка логирования:
    - Файл: DEBUG, подробный формат (время, модуль, файл:строка).
    - Консоль: INFO, краткий формат.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.DEBUG)

    # Убедиться, что каталог существует
    Path(LOG_FILE_PATH).parent.mkdir(parents=True, exist_ok=True)

    file_handler: Handler = RotatingFileHandler(
        LOG_FILE_PATH, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(Formatter(
        "%(asctime)s [%(levelname)s] %(name)s %(filename)s:%(lineno)d - %(message)s"
    ))

    console_handler: Handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter("%(levelname)s - %(message)s"))

    root.addHandler(file_handler)
    root.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """Возвращает именованный логгер."""
    return logging.getLogger(name)
