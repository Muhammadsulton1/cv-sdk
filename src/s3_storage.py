"""
Менеджер для работы с распределенной файловой системой SeaweedFS.

Класс предоставляет асинхронные методы для загрузки и скачивания изображений через SeaweedFS
с использованием протокола HTTP. Реализует автоматическое управление соединениями через
контекстный менеджер и повторные попытки при сетевых ошибках.

Зависимости:
- Требует настройки переменных окружения:
  * master_url: адрес мастер-ноды SeaweedFS (без указания порта)
  * bucket_name: имя коллекции (аналог бакета S3)
  * volume_url: адрес volume-сервера
  * ttl_bucket: время жизни объекта в формате "Xm" (по умолчанию "5m")

Особенности:
- Использует асинхронную библиотеку aiohttp с оптимизированными настройками соединений
- Применяет повторные попытки операций (3 попытки по умолчанию)
- Автоматическое кодирование изображений в формат JPEG с качеством 85%
- Встроенная обработка ошибок с логированием через централизованный logger
"""

import io
import os
import time
import uuid
import asyncio

import aiohttp
import cv2
import numpy as np

from PIL import Image

from src.data_class import S3Data
from utils.decorators import retry
from utils.err import S3UploadError, S3DownloadError, FrameDecodeError
from utils.logger import logger


class SeaweedFSManager:
    """
        Асинхронный менеджер для взаимодействия с SeaweedFS.

        Класс управляет загрузкой и скачиванием изображений, обеспечивая:
        - Автоматическое получение FID через мастер-ноду
        - Оптимизированное кодирование изображений
        - Управление соединениями через aiohttp
        - Повторные попытки при временных ошибках

        Пример использования:
            async with SeaweedFSManager() as fs:
                s3_data = await fs.upload_object(frame)
                image = await fs.download_object(s3_data.file_url)
    """

    def __init__(self) -> None:
        """
        Инициализация менеджера SeaweedFS.

        Загружает параметры подключения из переменных окружения.

        Атрибуты:
            master_url (str): Хост мастер-ноды SeaweedFS (без протокола и порта)
            bucket_name (str): Имя коллекции (аналог бакета S3)
            volume_url (str): Хост volume-сервера
            ttl (str): Время жизни объекта (формат '5m' для 5 минут)
            _session (aiohttp.ClientSession): Асинхронная сессия (инициализируется при входе в контекст)
        """
        self.master_url = os.getenv("master_url")
        self.bucket_name = os.getenv("bucket_name")
        self.volume_url = os.getenv("volume_url")
        self.ttl = os.getenv("ttl_bucket", "5m")
        self._session = None

    async def __aenter__(self):
        """
        Инициализация асинхронного контекстного менеджера.

        Создает клиентскую сессию aiohttp с оптимизированными параметрами:
        - limit_per_host=100: ограничение соединений на один хост
        - ttl_dns_cache=300: кэширование DNS на 300 секунд
        - keepalive_timeout=30: таймаут keep-alive соединений
        - Таймауты запросов: total=30 сек, connect=10 сек

        Возвращает:
            SeaweedFSManager: Экземпляр менеджера с активной сессией
        """
        connector = aiohttp.TCPConnector(
            limit_per_host=100,
            limit=200,
            ttl_dns_cache=300,
            use_dns_cache=True,
            force_close=True,
            enable_cleanup_closed=True
        )

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30, connect=10),
            connector=connector,
        )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Очистка ресурсов при выходе из контекста.

       Корректно закрывает HTTP-сессию, если она была создана.

       Аргументы:
           exc_type: Тип исключения (если возникло)
           exc_val: Значение исключения
           exc_tb: Объект трейсбэка
       """
        if self._session and not self._session.closed:
            await self._session.close()

    @retry(retries=3)
    async def upload_object(self, frame: np.ndarray) -> S3Data:
        """
        Загрузка изображения в SeaweedFS.

        Выполняет следующие операции:
        1. Получает уникальный FID от мастер-ноды
        2. Кодирует кадр в JPEG (качество 85%)
        3. Загружает данные на volume-сервер
        4. Формирует метаданные успешной загрузки

        Аргументы:
            frame (np.ndarray): Изображение в формате массива OpenCV (BGR)

        Возвращает:
            S3Data: Объект с метаданными загруженного файла, содержащий:
                - file_url: Публичный URL файла
                - file_id: Уникальный идентификатор файла (FID)
                - size: Размер файла в байтах

        Исключения:
            S3UploadError: При отсутствии FID или других критических ошибках
            FrameDecodeError: При неудачном кодировании изображения
            aiohttp.ClientError: При сетевых проблемах (автоматические повторы)
            asyncio.TimeoutError: При превышении таймаута соединения
        """
        try:
            assign_url = f"http://{self.master_url}:9333/dir/assign"
            params = {
                'ttl': self.ttl,
                'collection': self.bucket_name,
                'replication': '000'
            }

            async with self._session.get(assign_url, params=params, raise_for_status=True) as resp:
                assign_data = await resp.json()
                fid = assign_data.get("fid")
                if not fid:
                    raise S3UploadError("Не получен FID от master")

            upload_url = f"http://{self.volume_url}:8888/{self.bucket_name}/{fid}?ttl={self.ttl}"

            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]

            success, buf = cv2.imencode(".jpg", frame, encode_params)
            if not success:
                logger.error(f"Ошибка декодирования кадра в байты для загрузки в S3")
                raise FrameDecodeError

            image_bytes = buf.tobytes()

            async with self._session.put(
                    upload_url,
                    data=image_bytes,
                    headers={"Content-Type": "image/jpeg"},
                    raise_for_status=True
            ) as resp:
                pass

            return S3Data(
                file_url=f"http://{self.volume_url}:8888/{self.bucket_name}/{fid}",
                file_id=fid,
                size=len(buf.tobytes())
            )

        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError) as err:
            logger.error(f"Сетевая ошибка при загрузки данных в S3: {type(err).__name__} - {str(err)}")
            raise err
        except S3UploadError as err:
            logger.error(f"Ошибка загрузки данных в S3: {type(err).__name__} - {str(err)}")
            raise err

    @retry(retries=3)
    async def download_object(self, object_url: str):
        """
        Скачивание изображения из SeaweedFS.

        Аргументы:
            object_url (str): Полный URL объекта (полученный через upload_object.file_url)

        Возвращает:
            PIL.Image.Image: Загруженное изображение в объекте PIL

        Исключения:
            S3DownloadError: При критических ошибках обработки изображения
            aiohttp.ClientError: При сетевых проблемах (автоматические повторы)
            asyncio.TimeoutError: При превышении таймаута соединения
            OSError: При ошибках обработки изображения PIL
        """
        try:
            async with self._session.get(object_url, raise_for_status=True) as response:
                image_data = await response.read()
                return Image.open(io.BytesIO(image_data))

        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as err:
            logger.error(f"Сетевая ошибка при скачивании данных из S3: {type(err).__name__} - {str(err)}")
            raise err
        except S3DownloadError as err:
            logger.error(f"Ошибка скачивания данных из S3: {type(err).__name__} - {str(err)}")
            raise err
