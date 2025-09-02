import asyncio
import json
import os
import nats

from abc import abstractmethod, ABC
from src.data_scheme import S3Data
from src.s3_storage import SeaweedFSManager
from src.singeleton.connection_singeleton import NatsClient
from src.singeleton.yaml_reader import YamlReader
from utils.logger import logger

from nats.errors import ConnectionClosedError, TimeoutError
from nats.aio.errors import ErrConnectionClosed, ErrNoServers, ErrTimeout
from utils.err import NatsError


class AbstractReaderManager(ABC):
    def __init__(self):
        """Инициализирует компоненты системы."""
        self.reader_type = os.environ.get("reader_type", "opencv")
        if self.reader_type is None:
            raise ValueError("Пропущен аргумент 'reader_type' для выбора типа чтения кадров")

        self.uploader = None
        self.nats_url = os.getenv("nats_host", "nats://localhost:4222")

        self.setup_config = YamlReader()
        self.topic = self.setup_config.get('DataReader')['out_channel']

        self.nats_cli = None

    async def __aenter__(self):
        """Инициализация всех ресурсов при старте"""
        self.uploader = await SeaweedFSManager().__aenter__()
        self.nats_cli = await NatsClient.connect()
        return self

    async def __aexit__(self, *args):
        """Единая точка очистки"""
        if self.uploader:
            await self.uploader.__aexit__(*args)

        await NatsClient.close()

    async def publish_to_nats(self, message: S3Data) -> None:
        """Публикует сообщение в NATS с измерением времени.

        Args:
            message: Данные для публикации в формате JSON
        """
        try:
            await self.nats_cli.publish(self.topic, json.dumps({'file_url': message.file_url, 'fid': message.file_id},
                                                               ensure_ascii=False).encode('utf-8'))
        except Exception as e:
            logger.error(f"Ошибка публикации в NATS: {e}")
            raise

    # @abstractmethod
    # def process_frames(self, frames):
    #     pass

    @abstractmethod
    async def upload_frames(self, send_type, frames) -> S3Data:
        pass

    @abstractmethod
    async def runner(self) -> None:
        pass

    def process(self) -> None:
        async def _run():
            async with self:
                await self.runner()

        asyncio.run(_run())