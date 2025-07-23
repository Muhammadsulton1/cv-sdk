import logging
import os
from logging.handlers import TimedRotatingFileHandler

logs_dir = os.path.abspath(os.path.join(os.getcwd(), 'logs'))
os.makedirs(logs_dir, exist_ok=True)

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s: %(message)s')

file_handler = TimedRotatingFileHandler(
    filename=os.path.join(logs_dir, 'system.log'),
    when='midnight',
    interval=1,
    backupCount=7,
    encoding='utf-8'
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
    handler.close()

root_logger.addHandler(file_handler)

# Опционально: отключаем логирование от сторонних библиотек
logging.getLogger("urllib3").propagate = False
logging.getLogger("requests").propagate = False
