import logging
import os
import sys
import atexit
from concurrent_log_handler import ConcurrentRotatingFileHandler

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "processor.log")

logger = logging.getLogger("logs")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '%(asctime)s | %(name)s | %(levelname)s: %(message)s'
)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

file_handler = ConcurrentRotatingFileHandler(
    LOG_FILE,
    maxBytes=20 * 1024 * 1024,  # 20 МБ
    backupCount=5,
    encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)


def close_handlers():
    for handler in logger.handlers:
        handler.close()


atexit.register(close_handlers)


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Необработанное исключение",
                 exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception
