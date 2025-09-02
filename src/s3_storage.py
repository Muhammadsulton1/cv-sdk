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
import concurrent.futures
import io
import os
import asyncio
from typing import List

import aiohttp
import cv2
import numpy as np

from PIL import Image

from src.data_scheme import FrameData, S3Data
from src.singeleton.connection_singeleton import S3Client
from utils.decorators import retry_async, measure_latency_async
from utils.err import S3UploadError, S3DownloadError, FrameDecodeError, ArchiveDecodeError
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
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

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
        self._session = await S3Client.connect()

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
        await S3Client.close()

        self._executor.shutdown()

    async def _compress_frame(self, frame: np.ndarray) -> bytes:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        )

    async def _compress_frames_parallel(self, frames: List[np.ndarray]) -> bytes:
        tasks = [self._compress_frame(frame) for frame in frames]
        compressed_frames = await asyncio.gather(*tasks)

        return b''.join([len(f).to_bytes(4, 'big') + f for f in compressed_frames])

    async def _assign_fid(self) -> str:
        assign_url = f"http://{self.master_url}:9333/dir/assign"
        params = {'ttl': self.ttl, 'collection': self.bucket_name, 'replication': '000'}

        async with self._session.get(assign_url, params=params, raise_for_status=True) as resp:
            assign_data = await resp.json()
            if fid := assign_data.get("fid"):
                return fid
        raise S3UploadError("Не получен FID от master")

    @retry_async(retries=3)
    async def upload_object(self, frames_batch: FrameData):
        source_name = frames_batch.cam_source

        if not frames_batch.frames:
            raise ValueError("No frames to upload")

        if len(frames_batch.frames) == 1:
            frame = frames_batch.frames[0]
            if isinstance(frame, np.ndarray):
                loop = asyncio.get_event_loop()
                frame_bytes = await loop.run_in_executor(
                    self._executor,
                    lambda: cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
                )
            else:
                frame_bytes = frame
            content_type = "image/jpeg"
            upload_file = io.BytesIO(frame_bytes)
        else:
            compressed_data = await self._compress_frames_parallel(frames_batch.frames)
            content_type = "application/octet-stream"
            upload_file = io.BytesIO(compressed_data)

        upload_file.seek(0)

        fid = await self._assign_fid()
        upload_url = f"http://{self.volume_url}:8888/{self.bucket_name}/{source_name}/{fid}?ttl={self.ttl}"

        async with self._session.put(
                upload_url,
                data=upload_file,
                headers={"Content-Type": content_type},
                raise_for_status=True
        ) as resp:
            pass

        return S3Data(
            file_url=upload_url,
            file_id=fid,
            content_type=content_type
        )