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
from typing import List, Union

import cv2
import numpy as np

from src.data_scheme import FrameData, S3Data
from src.singeleton.connection_singeleton import S3Client
from utils.decorators import retry_async, measure_latency_async
from utils.err import S3UploadError, S3DownloadError, S3FrameDecodeError, S3GetError
from utils.logger import logger


class SeaweedFSManager:
    """
        Менеджер для работы с распределённой файловой системой SeaweedFS.

        Обеспечивает асинхронную загрузку и скачивание изображений через HTTP.
        Реализует:
            - Автоматическое получение FID от master-ноды SeaweedFS.
            - Оптимизированное кодирование изображений в JPEG с качеством 85%.
            - Параллельное сжатие пакетов кадров.
            - Управление HTTP-сессией через контекстный менеджер.
            - Повторные попытки операций при сетевых ошибках.
            - Измерение задержек операций через декоратор measure_latency_async.
        Зависимости (переменные окружения):
            master_url — хост мастер-ноды (без порта);
            bucket_name — имя коллекции (аналог S3 bucket);
            volume_url — хост volume-сервера;
            ttl_bucket — время жизни объекта (формат "Xm", по умолчанию "5m").
        """

    def __init__(self) -> None:
        """
            Инициализация менеджера.

            Загружает настройки из переменных окружения и создаёт пул потоков
            для операций кодирования и декодирования изображений.
            Атрибуты:
                master_url (str): адрес master-ноды SeaweedFS.
                bucket_name (str): имя коллекции для хранения объектов.
                volume_url (str): адрес volume-сервера для операций PUT/GET.
                ttl (str): время жизни загружаемых объектов.
                _session: HTTP-сессия aiohttp (инициализируется в __aenter__).
                _executor: ThreadPoolExecutor для выполнения CPU-bound задач.
        """
        self.master_url = os.getenv("master_url")
        self.bucket_name = os.getenv("bucket_name")
        self.volume_url = os.getenv("volume_url")
        self.ttl = os.getenv("ttl_bucket", "5m")
        self._session = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    async def __aenter__(self):
        """
            Вход в асинхронный контекстный менеджер.

            Устанавливает соединение с SeaweedFS через S3Client
            (aiohttp.ClientSession с оптимизированными параметрами).
            Возвращает:
                SeaweedFSManager: экземпляр менеджера с открытой сессией.
        """
        self._session = await S3Client.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
            Выход из асинхронного контекста менеджера.

            Закрывает HTTP-сессию и завершает пул потоков.
        """
        await S3Client.close()

        self._executor.shutdown()

    async def _compress_frame(self, frame: np.ndarray) -> bytes:
        """
        Сжимает один кадр в JPEG в отдельном потоке.
        Параметры:
            frame (np.ndarray): изображение в формате BGR.
        Возвращает:
            bytes: байтовое представление JPEG-изображения.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        )

    async def _compress_frames_parallel(self, frames: List[np.ndarray]) -> bytes:
        """
       Параллельно сжимает список кадров и объединяет результаты.
       Каждый кадр оборачивается префиксом длины (4 байта big-endian).
       Параметры:
           frames (List[np.ndarray]): список кадров для сжатия.
       Возвращает:
           bytes: объединённые байты всех JPEG-кадров с метками длины.
       """
        tasks = [self._compress_frame(frame) for frame in frames]
        compressed_frames = await asyncio.gather(*tasks)

        return b''.join([len(f).to_bytes(4, 'big') + f for f in compressed_frames])

    async def _assign_fid(self) -> str:
        """
        Запрашивает у master-ноды свободный FID для загрузки.
        Формирует GET-запрос:
            http://{master_url}:9333/dir/assign?ttl={ttl}&collection={bucket_name}&replication=000
        Возвращает:
            str: полученный FID.
        Исключения:
            S3GetError: если FID не возвращён в ответе.
        """
        assign_url = f"http://{self.master_url}:9333/dir/assign"
        params = {'ttl': self.ttl, 'collection': self.bucket_name, 'replication': '000'}

        async with self._session.get(assign_url, params=params, raise_for_status=True) as resp:
            assign_data = await resp.json()
            if fid := assign_data.get("fid"):
                return fid
        raise S3GetError("Не получен FID от master")

    @retry_async(retries=3)
    async def upload_object(self, frames_batch: FrameData):
        """
        Загружает один или несколько кадров в SeaweedFS.

        Параметры:
            frames_batch (FrameData): структура с полями:
                cam_source (str)  — идентификатор источника;
                frames (List[np.ndarray] | List[bytes]) — кадры или байты.
        Логика:
            - Если один кадр: сжатие в JPEG или передача байтов.
            - Если несколько: вызов _compress_frames_parallel.
            - Запрос FID, формирование URL PUT-запроса.
            - Отправка данных на volume-сервер.
        Возвращает:
            S3Data: информация об объекте (URL, FID, content_type).
        Исключения:
            ValueError     — если нет кадров для загрузки.
            S3UploadError  — при ошибке получения FID.
            HTTPError      — при ошибке PUT-запроса.
        """
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

    @measure_latency_async()
    async def download_object(self, message: S3Data):
        """
        Скачивает и декодирует изображение(я) из SeaweedFS.

        Параметры:
            message (S3Data): объект с полями:
                file_url     — полный URL ресурса;
                content_type — тип содержимого.
        Логика:
            - GET-запрос к message.file_url.
            - Чтение всего контента.
            - Вызов _decode_content для преобразования байт в np.ndarray.
        Возвращает:
            np.ndarray при одиночном JPEG или List[np.ndarray] для пакетов.
        Исключения:
            S3DownloadError — при ошибке HTTP-запроса.
        """
        try:
            async with self._session.get(message.file_url, raise_for_status=True) as response:
                content = await response.read()
            return await self._decode_content(content, message.content_type)
        except S3DownloadError:
            raise S3DownloadError

    @measure_latency_async()
    async def _decode_content(self, content: bytes, content_type: str) -> Union[np.ndarray, List[np.ndarray]]:
        """
            Декодирует бинарные данные в изображение(я) используя ThreadPool.

            Параметры:
                content (bytes): поток байт из HTTP-ответа.
                content_type (str): "image/jpeg" или "application/octet-stream".
            Логика:
                - Для JPEG: однокадровый cv2.imdecode.
                - Для бинарного пакета: чтение длины (4 байта) + кадр + cv2.imdecode.
            Возвращает:
                np.ndarray или List[np.ndarray] в зависимости от content_type.
            Исключения:
                S3FrameDecodeError — при ошибке декодирования кадра.
        """
        try:
            loop = asyncio.get_event_loop()

            if content_type == "image/jpeg":
                return await loop.run_in_executor(
                    self._executor,
                    lambda: cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
                )

            elif content_type == "application/octet-stream":
                frames = []
                offset = 0

                while offset < len(content):
                    length = int.from_bytes(content[offset:offset + 4], 'big')
                    offset += 4

                    frame_data = content[offset:offset + length]
                    offset += length

                    frame = await loop.run_in_executor(
                        self._executor,
                        lambda: cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    )
                    frames.append(frame)

                return frames

        except S3FrameDecodeError:
            raise S3FrameDecodeError
