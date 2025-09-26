import asyncio
import json
import os
from dataclasses import asdict

from abc import abstractmethod, ABC
from src.data_scheme import S3Data
from src.s3_storage import SeaweedFSManager
from src.singeleton.connection_singeleton import NatsClient
from src.singeleton.yaml_reader import YamlReader
from utils.logger import logger


class AbstractReaderManager(ABC):
    def __init__(self):
        """Инициализирует компоненты системы."""
        self.reader_type = os.environ.get("reader_type", "opencv")
        if self.reader_type is None:
            raise ValueError("Пропущен аргумент 'reader_type' для выбора типа чтения кадров")

        self.s3_cli = None
        self.nats_url = os.getenv("nats_host", "nats://localhost:4222")

        self.setup_config = YamlReader()
        self.topic = self.setup_config.get('DataReader')['out_channel']

        self.nats_cli = None

    async def __aenter__(self):
        """Инициализация всех ресурсов при старте"""
        self.s3_cli = await SeaweedFSManager().__aenter__()
        self.nats_cli = await NatsClient.connect()
        return self

    async def __aexit__(self, *args):
        """Единая точка очистки"""
        if self.s3_cli:
            await self.s3_cli.__aexit__(*args)

        await NatsClient.close()

    async def publish_to_nats(self, message: S3Data) -> None:
        """Публикует сообщение в NATS с измерением времени.

        Args:
            message: Данные для публикации в формате JSON
        """
        try:
            await self.nats_cli.publish(self.topic, json.dumps(asdict(message)).encode('utf-8'))
        except Exception as e:
            logger.error(f"Ошибка публикации в NATS: {e}")
            raise

    @abstractmethod
    async def runner(self) -> None:
        pass

    def process(self) -> None:
        async def _run():
            async with self:
                await self.runner()

        asyncio.run(_run())
