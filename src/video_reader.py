import asyncio
import json
import os
import uuid
import aiohttp
import av
import cv2
import nats

from nats.errors import ConnectionClosedError, TimeoutError
from nats.aio.errors import ErrConnectionClosed, ErrNoServers, ErrTimeout
from abs_src.abs_reader import AbstractReader, FileUploader
from utils.logger import logger


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


class SeaweedFSUploader(FileUploader):
    def __init__(self):
        self.master_url = os.getenv("master_url")
        self.volume_url = os.getenv("volume_url")
        self.bucket_name = os.getenv("bucket_name")
        self.ttl = os.getenv("ttl_bucket")
        self._session = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._session:
            await self._session.close()

    async def upload_file(self, file_data: bytes, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                return await self.upload(file_data)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
                logger.error(f"Ошибка при загрузки кадра в SEAWEADFS: {e}")
                logger.error(f"Попытка {attempt + 1} из {max_retries}")

    async def upload(self, file_data: bytes) -> str:
        try:
            assign_url = f"{self.master_url}/dir/assign?ttl={self.ttl}"
            async with self._session.get(assign_url) as resp:
                assign_data = await resp.json()
                fid = assign_data["fid"]

            upload_url = f"{self.volume_url}/{self.bucket_name}/{fid}?ttl={self.ttl}"
            file_name = f"{uuid.uuid4()}.jpg"

            form_data = aiohttp.FormData()
            form_data.add_field(
                'file',
                file_data,
                filename=file_name,
                content_type='image/jpeg'
            )

            async with self._session.post(upload_url, data=form_data) as resp:
                resp.raise_for_status()

            return f"{self.volume_url}/{self.bucket_name}/{fid}"
        except (aiohttp.ClientError, aiohttp.ClientConnectorError, asyncio.TimeoutError) as err:
            logger.error(f"Ошибка подключения при загрузки кадра в SEAWEADFS: {err}")
            raise err
        except Exception as err:
            logger.error(f"Ошибка при загрузки кадра в SEAWEADFS: {err}")
            raise err


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
        self.uploader = SeaweedFSUploader()
        self.nats_url = os.getenv("nats_host", "nats://localhost:4222")
        self.topic = os.getenv("topic_stream")

    async def main(self):
        nc = None
        try:
            nc = await nats.connect(
                servers=[self.nats_url],
                connect_timeout=5,
                max_reconnect_attempts=3
            )
            logger.info("Подключения к NATS успешно")

        except (ErrConnectionClosed, ErrNoServers, ErrTimeout, ConnectionClosedError, TimeoutError) as err:
            logger.error(f"NATS соединение упало с ошибкой: {err}")
            return

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

                    _, buffer = cv2.imencode('.jpg', frame)
                    image_data = buffer.tobytes()

                    async with self.uploader as publisher:
                        try:
                            file_url = await publisher.upload_file(image_data)

                            message = {
                                "frame_id": f"frame_{frame_number:06d}",
                                "seaweed_url": file_url,
                                "timestamp": frame_timestamp,
                                "meta": meta_info
                            }

                            json_message = json.dumps(message).encode('utf-8')

                            await nc.publish(self.topic, json_message)
                            logger.debug(f"Отправлен кадр {frame_number}: {message}")

                        except (ConnectionClosedError, TimeoutError) as e:
                            logger.error(f"Ошибка обработки кадра {frame_number}: {e}")

        except KeyboardInterrupt:
            logger.info("Процесс обработки остановлен пользователем")
        except Exception as e:
            logger.exception(f"Не предвиденная ошибка обработки события в VideoReader", {e})
        finally:
            if nc and not nc.is_closed:
                await nc.drain()
                await nc.close()
                nc = None
                logger.info("NATS соединение закрыто")

    def process(self):
        asyncio.run(self.main())
