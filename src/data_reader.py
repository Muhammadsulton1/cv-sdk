import asyncio
import json
import os
from typing import Dict, Any

import av
import cv2
import nats

from nats.errors import ConnectionClosedError, TimeoutError
from nats.aio.errors import ErrConnectionClosed, ErrNoServers, ErrTimeout

from abs_src.abs_reader import AbstractReader
from src.data_class import FrameData
from src.s3_storage import SeaweedFSManager
from src.singeleton.yaml_reader import YamlReader
from utils.err import NatsError
from utils.logger import logger
from utils.decorators import measure_latency_sync, measure_latency_async


class AVReader(AbstractReader):
    """Реализация чтения видео через библиотеку PyAV."""

    def __init__(self):
        """Инициализирует контейнер, видеопоток и счетчик кадров."""
        super().__init__()
        self.container = None
        self.video_stream = None
        self.frame_generator = None
        self._frames_processed = 0

    def __enter__(self):
        """Поддержка контекстного менеджера (with)."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Обеспечивает закрытие ресурсов при выходе из контекста."""
        self.close()

    def open(self):
        """
            Открывает видео источник и инициализирует поток для чтения.

            Raises:
                RuntimeError: Если источник не удалось открыть.
        """
        self.container = av.open(
            self.source,
            options={"rtsp_flags": "prefer_tcp"}
        )

        if self.container is None:
            raise RuntimeError(f"Ошибка открытия источника: {self.source}")

        video_stream = self.container.streams.video[self.stream_index]
        self.video_stream = video_stream

        self.codec_name = video_stream.codec_context.name
        self.width = video_stream.codec_context.width
        self.height = video_stream.codec_context.height
        self.framerate = float(video_stream.average_rate)
        self._frames_processed = 0

        # Инициализация генератора кадров
        self.frame_generator = self._generate_frames()

    def close(self):
        """Закрывает контейнер с видео и освобождает ресурсы."""
        if self.container:
            self.container.close()

    @measure_latency_sync()
    def _generate_frames(self):
        """Приватный генератор кадров.

        Yields:
            FrameData: Объект с данными кадра и метаинформацией.
        """
        frame_count = 0
        for packet in self.container.demux(self.video_stream):
            for frame in packet.decode():
                frame_count += 1
                if self.skip_frames > 0 and frame_count % (self.skip_frames + 1) != 0:
                    continue

                frame_number = frame_count
                frame_time = float(frame.pts * frame.time_base)

                rgb_frame = frame.to_ndarray(format="bgr24")

                yield FrameData(image=rgb_frame,
                                number=frame_number,
                                timestamp=frame_time)
                self._frames_processed += 1

    @measure_latency_sync()
    def get_frame(self) -> FrameData | None:
        """
        Возвращает следующий кадр как изображение.

        Returns:
            FrameData | None: Данные кадра или None при завершении потока.
        """
        try:
            return next(self.frame_generator)
        except StopIteration:
            return None

    def __iter__(self):
        """Возвращает итератор для поддержки цикла for."""
        return self

    def __next__(self):
        """
        Возвращает следующий кадр через протокол итератора.

        Raises:
            StopIteration: Когда кадры закончились.
        """
        frame = self.get_frame()
        if frame is None:
            raise StopIteration
        return frame

    @property
    def frames_processed(self):
        """Возвращает количество обработанных кадров"""
        return self._frames_processed

    @property
    def info(self):
        """
        Возвращает метаинформацию о видео потоке.

        Returns:
            dict: Словарь с параметрами видео.
        """
        return {
            "source": self.source,
            "codec": self.codec_name,
            "width": self.width,
            "height": self.height,
            "framerate": self.framerate,
            "frames_skipped": self.skip_frames,
            "frames_processed": self.frames_processed
        }


class OpencvVideoReader(AbstractReader):
    """Реализация чтения видео через OpenCV."""

    def __init__(self):
        """Инициализирует параметры видео и счетчик кадров."""
        super().__init__()
        self.codec_name = "unknown"
        self.width = 0
        self.height = 0
        self.framerate = 0
        self.frames_processed = 0

    def __enter__(self):
        """Поддержка контекстного менеджера (with)."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Обеспечивает закрытие ресурсов при выходе из контекста."""
        self.close()

    def open(self):
        """
            Открывает видео источник через OpenCV.

            Raises:
                RuntimeError: Если источник не удалось открыть.
        """
        self.video_stream = cv2.VideoCapture(self.source)
        if not self.video_stream.isOpened():
            raise RuntimeError(f"Ошибка открытия источника: {self.source}")

        # Получаем свойства видео
        self.width = int(self.video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.framerate = self.video_stream.get(cv2.CAP_PROP_FPS)

        fourcc_int = self.video_stream.get(cv2.CAP_PROP_FOURCC)
        if fourcc_int != 0:
            self.codec_name = "".join([
                chr(int(fourcc_int) >> 8 * i & 0xFF)
                for i in range(4)
            ])

        self.frames_processed = 0

    def close(self):
        """Освобождает ресурсы видеозахвата."""
        if self.video_stream is not None and self.video_stream.isOpened():
            self.video_stream.release()
        self.video_stream = None

    def get_frame(self) -> FrameData | None:
        """Возвращает следующий кадр с учетом skip_frames.

        Returns:
            FrameData | None: Данные кадра или None при завершении потока.
        """
        for _ in range(self.skip_frames):
            if not self.video_stream.grab():
                return None
            self.frames_processed += 1

        ret, frame = self.video_stream.read()
        self.frames_processed += 1

        if not ret:
            return None

        frame_number = self.frames_processed
        frame_time = self.video_stream.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Переводим в секунд
        self.frames_processed += 1

        return FrameData(image=frame,
                         number=frame_number,
                         timestamp=frame_time)

    def __iter__(self):
        """Возвращает итератор для поддержки цикла for."""
        return self

    def __next__(self):
        """Возвращает следующий кадр через протокол итератора.

        Raises:
            StopIteration: Когда кадры закончились.
        """
        frame = self.get_frame()
        if frame is None:
            raise StopIteration
        return frame

    @property
    def info(self):
        """
        Возвращает метаинформацию о видео потоке.

        Returns:
            dict: Словарь с параметрами видео.
        """
        return {
            "source": self.source,
            "codec": self.codec_name,
            "width": self.width,
            "height": self.height,
            "framerate": self.framerate,
            "frames_skipped": self.skip_frames,
            "frames_processed": self.frames_processed
        }


class VideoReaderFactory:
    """"Фабрика для создания объектов чтения видео."""

    _readers = {
        "opencv": OpencvVideoReader,
        "av": AVReader,
    }

    @classmethod
    def create_reader(cls, reader_type: str):
        """
        Создает экземпляр видео ридера указанного типа.

        Args:
            reader_type: Тип ридера ('opencv' или 'av')

        Returns:
            AbstractReader: Экземпляр класса ридера

        Raises:
            ValueError: Если указан неподдерживаемый тип ридера
        """
        reader_class = cls._readers.get(reader_type.lower())

        if reader_class is None:
            supported = ", ".join(cls._readers.keys())
            raise ValueError(
                f"Не поддерживаемый источник для чтения видео потока: '{reader_type}'. "
                f"поддерживаемые типы: {supported}"
            )

        return reader_class()

    @classmethod
    def register_reader(cls, reader_type: str, reader_class):
        """
        Регистрирует новый тип ридера в фабрике.

        Args:
            reader_type: Идентификатор типа ридера
            reader_class: Класс ридера (должен быть вызываемым)

        Raises:
            TypeError: Если переданный класс не вызываемый
        """
        if not callable(reader_class):
            raise TypeError("VideoReader class должен быть вызываемым объектом")
        cls._readers[reader_type.lower()] = reader_class

    @classmethod
    def supported_readers(cls):
        """"Возвращает список поддерживаемых типов ридеров."""
        return list(cls._readers.keys())


class ReaderManager:
    """Управляет процессом чтения видео, обработкой кадров и публикацией результатов."""
    def __init__(self):
        """Инициализирует компоненты системы."""
        reader_type = os.environ.get("reader_type", "opencv")
        if reader_type is None:
            raise ValueError("Пропущен аргумент 'reader_type' для выбора типа чтения кадров")

        self.reader = VideoReaderFactory.create_reader(reader_type)
        # self.uploader = SeaweedFSManager()
        self.uploader = None
        self.nats_url = os.getenv("nats_host", "nats://localhost:4222")

        self.setup_config = YamlReader()
        self.topic = self.setup_config.get('DataReader')['out_channel']

        self.nats_cli = None

    async def __aenter__(self):
        """Инициализация всех ресурсов при старте"""
        self.uploader = await SeaweedFSManager().__aenter__()
        await self.ensure_nats_connected()
        return self

    async def __aexit__(self, *args):
        """Единая точка очистки"""
        if self.uploader:
            await self.uploader.__aexit__(*args)
        await self.close()  # Закрытие NATS

    async def ensure_nats_connected(self):
        """
        Устанавливает подключение к серверу NATS при необходимости.

       Raises:
           NatsError: При неудачном подключении.
       """
        if self.nats_cli is None or self.nats_cli.is_closed:
            try:
                self.nats_cli = await nats.connect(
                    servers=[self.nats_url],
                    connect_timeout=5,
                    max_reconnect_attempts=3
                )
                logger.info("Подключение к NATS успешно установлено")
            except (ErrConnectionClosed, ErrNoServers,
                    ErrTimeout, ConnectionClosedError,
                    TimeoutError) as err:
                logger.error(f"Ошибка подключения к NATS: {err}")
                raise NatsError

    @measure_latency_async()
    async def process_frame(self, frame_data: FrameData) -> Dict[str, Any]:
        """
        Обрабатывает кадр: загружает в SeaweedFS и формирует сообщение.

        Args:
            frame_data: Данные обрабатываемого кадра

        Returns:
            dict: Сообщение для публикации в NATS

        Raises:
            Exception: При ошибках обработки кадра
        """
        try:
            # async with self.uploader as uploader:
            upload_result = await self.uploader.upload_object(frame_data.image)
            # upload_result = await uploader.upload_object(frame_data.image)

            message = {
                "frame_id": f"frame_{frame_data.number:06d}",
                "seaweed_url": upload_result.file_url,
                "file_id": upload_result.file_id,
                "timestamp": frame_data.timestamp,
                "size": upload_result.size,
                "meta": {
                    "width": frame_data.image.shape[1],
                    "height": frame_data.image.shape[0],
                    "channels": frame_data.image.shape[2]
                }
            }

            return message

        except Exception as e:
            logger.error(f"Ошибка обработки кадра {frame_data.number}: {e}")
            raise

    async def publish_to_nats(self, message: Dict[str, Any]) -> None:
        """Публикует сообщение в NATS с измерением времени.

        Args:
            message: Данные для публикации в формате JSON
        """
        try:
            json_message = json.dumps(message, ensure_ascii=False).encode('utf-8')
            await self.nats_cli.publish(self.topic, json_message)
        except Exception as e:
            logger.error(f"Ошибка публикации в NATS: {e}")
            raise

    async def runner(self) -> None:
        """Основной цикл обработки видео"""
        # await self.ensure_nats_connected()

        try:
            with self.reader as stream:
                while True:
                    try:
                        frame_data = stream.get_frame()
                        if frame_data is None:
                            logger.info("Конец видеопоток")
                            break

                        message = await asyncio.create_task(self.process_frame(frame_data))

                        await self.publish_to_nats(message)
                        logger.debug(f"Обработан кадр {frame_data.number}")

                    except Exception as e:
                        logger.error(f"Ошибка в цикле обработки: {e}", exc_info=True)
                        continue

        except KeyboardInterrupt:
            logger.info("Остановка по запросу пользователя")
        except Exception as e:
            logger.exception(f"Критическая ошибка в runner: {e}")
        finally:
            await self.close()

    async def close(self) -> None:
        """Корректно закрывает все ресурсы системы."""
        logger.info("Закрытие ReaderManager")

        if self.nats_cli and not self.nats_cli.is_closed:
            try:
                await self.nats_cli.drain()
                await self.nats_cli.close()
                logger.info("NATS соединение закрыто")
            except Exception as e:
                logger.error(f"Ошибка закрытия NATS: {e}")

    def process(self):
        async def _run():
            async with self:
                await self.runner()

        asyncio.run(_run())
        # asyncio.run(self.runner())
