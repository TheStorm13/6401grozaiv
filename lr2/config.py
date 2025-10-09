from pathlib import Path
# todo: может убрать final
from typing import Final

# Определение корня проекта (более надежный способ)
PROJECT_ROOT: Final[Path] = Path(__file__).parent.resolve()

# Пути к файлам и директориям
LOG_DIR: Final[Path] = PROJECT_ROOT / "logs"
LOG_FILE_PATH: Final[Path] = LOG_DIR / "app.log"
PHOTO_DIR: Final[Path] = PROJECT_ROOT / "images"
IMAGE_EXTENSIONS: Final[list[str]] = [".jpg", ".jpeg", ".png"]

# Создаем директорию для логов, если она не существует
LOG_DIR.mkdir(exist_ok=True)
PHOTO_DIR.mkdir(exist_ok=True)

# Ключ API для внешнего сервиса
API_KEY = "live_Pqi8F98HJlWQtYZ55C3X93u87ZO6cTlXz6o00zQOUcbOMPkUNr5t3ImzRs88LFca"
