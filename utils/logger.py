import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# === Фиксированная директория для логов ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, "processor.log")

# === Настройка логгера ===
logger = logging.getLogger("logs")
logger.setLevel(logging.DEBUG)

# Формат логов
formatter = logging.Formatter(
    '%(asctime)s | %(name)s | %(levelname)s: %(message)s'
)

# Консоль
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

# Файл с ротацией по размеру (20 МБ, до 5 старых файлов)
file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=20 * 1024 * 1024,  # 20 MB
    backupCount=5,
    encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Добавляем обработчики
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


# === Перехват всех необработанных исключений ===
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # стандартное поведение для Ctrl+C
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Необработанное исключение",
                 exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception
