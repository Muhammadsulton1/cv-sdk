import asyncio
import json
import os
import time
import uuid

import av
import cv2
import nats

from nats.errors import ConnectionClosedError, TimeoutError
from nats.aio.errors import ErrConnectionClosed, ErrNoServers, ErrTimeout

from abs_src.abs_reader import AbstractReader
from src.s3_storage import SeaweedFSManager
from src.singeleton.yaml_reader import YamlReader
from utils.err import NatsError
from utils.logger import logger
from utils.decorators import measure_latency_sync, measure_latency_async, measure_detailed_time


class AVReader(AbstractReader):
    def __init__(self):
        super().__init__()
        self.container = None
        self.video_stream = None
        self.frame_generator = None
        self._frames_processed = 0

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
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
        if self.container:
            self.container.close()

    @measure_latency_sync()
    def _generate_frames(self):
        """Приватный генератор кадров"""
        frame_count = 0
        for packet in self.container.demux(self.video_stream):
            for frame in packet.decode():
                frame_count += 1
                if self.skip_frames > 0 and frame_count % (self.skip_frames + 1) != 0:
                    continue

                frame_number = frame_count
                frame_time = float(frame.pts * frame.time_base)

                rgb_frame = frame.to_ndarray(format="bgr24")
                yield {
                    'image': rgb_frame,
                    'number': frame_number,
                    'timestamp': frame_time
                }
                self._frames_processed += 1

    @measure_latency_sync()
    def get_frame(self):
        """Возвращает следующий кадр как изображение"""
        try:
            return next(self.frame_generator)
        except StopIteration:
            return None

    def __iter__(self):
        return self

    def __next__(self):
        """Поддержка итерации"""
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
    def __init__(self):
        super().__init__()
        self.codec_name = "unknown"
        self.width = 0
        self.height = 0
        self.framerate = 0
        self.frames_processed = 0

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
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
        if self.video_stream is not None and self.video_stream.isOpened():
            self.video_stream.release()
        self.video_stream = None

    # @measure_latency_sync()
    def get_frame(self):
        """Возвращает следующий кадр с учетом skip_frames."""
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

        return {
            'image': frame,
            'number': frame_number,
            'timestamp': frame_time
        }

    def __iter__(self):
        return self

    def __next__(self):
        """Поддержка итерации."""
        frame = self.get_frame()
        if frame is None:
            raise StopIteration
        return frame

    @property
    def info(self):
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
    """Фабрика для создания объектов чтения видео."""

    _readers = {
        "opencv": OpencvVideoReader,
        "av": AVReader,
    }

    @classmethod
    def create_reader(cls, reader_type: str):
        """
        Создает экземпляр видео ридера указанного типа.

        :param reader_type: Тип ридера ('opencv' или 'av')
        :return: Экземпляр класса ридера
        :raises ValueError: Если указан неподдерживаемый тип ридера
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

        :param reader_type: Идентификатор типа ридера
        :param reader_class: Класс ридера (должен быть вызываемым)
        """
        if not callable(reader_class):
            raise TypeError("VideoReader class должен быть вызываемым объектом")
        cls._readers[reader_type.lower()] = reader_class

    @classmethod
    def supported_readers(cls):
        """Возвращает список поддерживаемых типов ридеров."""
        return list(cls._readers.keys())


class ReaderManager:
    def __init__(self):
        reader_type = os.environ.get("reader_type", "opencv")
        if reader_type is None:
            raise ValueError("Пропущен аргумент 'reader_type' для выбора типа чтения кадров")

        self.reader = VideoReaderFactory.create_reader(reader_type)
        self.uploader = SeaweedFSManager()
        self.nats_url = os.getenv("nats_host", "nats://localhost:4222")

        self.setup_config = YamlReader()
        self.topic = self.setup_config.get('DataReader')['out_channel']

        self.nats_cli = None

    async def ensure_nats_connected(self):
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

    async def runner(self):
        await self.ensure_nats_connected()
        try:
            with self.reader as stream:
                while True:
                    frame_data = stream.get_frame()
                    meta_info = stream.info
                    if frame_data is None:
                        logger.info("Конец видео стрима")
                        break

                    frame = frame_data['image']
                    frame_number = frame_data['number']
                    frame_timestamp = frame_data['timestamp']

                    async with self.uploader as publisher:
                        try:
                            file_url = await asyncio.create_task(publisher.upload_object(frame))

                            message = {
                                "frame_id": f"frame_{frame_number:06d}",
                                "seaweed_url": file_url,
                                "timestamp": frame_timestamp,
                                "meta": meta_info
                            }
                        except (ConnectionClosedError, TimeoutError) as e:
                            logger.error(f"Ошибка обработки кадра {frame_number}: {e}")
                            raise NatsError

                    json_message = json.dumps(message).encode('utf-8')

                    await self.nats_cli.publish(self.topic, json_message)
                    logger.debug(f"Отправлен кадр {frame_number}: {message}")

        except KeyboardInterrupt:
            logger.info("Процесс обработки остановлен пользователем")
        except Exception as e:
            logger.exception(f"Не предвиденная ошибка обработки события в VideoReader", {e})
            await self.close()
        finally:
            await self.close()

    async def close(self):
        logger.info(f'Инициализирую остановку и освобождения сервисов')

        if self.nats_cli and not self.nats_cli.is_closed:
            await self.nats_cli.drain()
            await self.nats_cli.close()
            self.nats_cli = None
            logger.info("NATS соединение закрыто")

    def process(self):
        asyncio.run(self.runner())
