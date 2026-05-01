import os
from abc import ABC, abstractmethod

import nats
from nats.js.api import StreamConfig, RetentionPolicy, StorageType

from common.singeleton.config_manager import ConfigManager
from common.singeleton.connection import NatsClient, S3Client
from common.utils.logger import logger


class AbstractStreamManager(ABC):
    """
    Базовый класс: всегда поднимает JetStream + S3.
    Ридер публикует кадры → S3 + ключ в JetStream.
    Инференс читает ключ из JetStream → скачивает кадр из S3.
    """

    JS_NAME: str = "FrameStream"

    def __init__(self) -> None:
        self._service_name = os.getenv("SERVICE_NAME")
        if not self._service_name:
            raise ValueError("SERVICE_NAME environment variable is not set")

        self._cfg = ConfigManager().config_service_name(self._service_name)
        if not self._cfg:
            raise ValueError(f"No config found for service '{self._service_name}'")

        self.topic_in_channel = self._cfg.get("in_channel")
        self.topic_out_channel = self._cfg.get("out_channel")
        self.js_ttl: int = int(os.getenv("STREAM_TTL", 300))

        self.js = None
        self._nats_client = None
        self._s3_raw_client = None

    def _publish_subject(self, cam_id: str) -> str:
        """
        Subject для публикации одного кадра.
        Формат: {out_channel}.{cam_id}
        Пример: video-stream.a3f9c1
        """
        if not self.topic_out_channel:
            raise ValueError(f"out_channel не задан для сервиса '{self._service_name}'")
        return f"{self.topic_out_channel}.{cam_id}"

    def _subscribe_subject(self) -> str:
        """
        Subject-фильтр для подписки.
        Формат: {in_channel}.>  — получаем кадры от всех камер канала.
        Пример: video-stream.>
        """
        if not self.topic_in_channel:
            raise ValueError(f"in_channel не задан для сервиса '{self._service_name}'")
        return f"{self.topic_in_channel}.>"

    def _stream_subjects(self) -> list[str]:
        """
        Subjects которые покрывает стрим.
        Собираем из обоих каналов — стрим должен знать про все входы.
        """
        subjects = set()
        if self.topic_out_channel:
            subjects.add(f"{self.topic_out_channel}.>")
        if self.topic_in_channel:
            subjects.add(f"{self.topic_in_channel}.>")
        if not subjects:
            raise ValueError(f"Не задан ни in_channel, ни out_channel для '{self._service_name}'")
        return list(subjects)

    async def __aenter__(self):
        self._nats_client = NatsClient()
        await self._nats_client.connect()
        self.js = await self._nats_client.get_jetstream()

        self._s3_raw_client = await S3Client.connect()

        await self._ensure_stream()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await S3Client.close()
        if self._nats_client is not None:
            await self._nats_client.close()

    async def _ensure_stream(self) -> None:
        """Создаёт или обновляет JetStream стрим."""
        subjects = self._stream_subjects()
        cfg = StreamConfig(
            name=self.JS_NAME,
            subjects=subjects,
            retention=RetentionPolicy.LIMITS,
            max_age=float(self.js_ttl),
            storage=StorageType.FILE,
        )
        try:
            await self.js.stream_info(cfg.name)
        except nats.js.errors.NotFoundError:
            try:
                await self.js.add_stream(cfg)
                logger.info(
                    "JetStream stream %r создан, subjects=%s",
                    cfg.name, subjects,
                )
            except nats.js.errors.BadRequestError:
                logger.error(
                    "Не удалось создать stream %r: subjects заняты. "
                    "Проверьте `nats stream ls`.",
                    cfg.name,
                )
                raise
        else:
            await self.js.update_stream(cfg)
            logger.info(
                "JetStream stream %r обновлён, subjects=%s",
                cfg.name, subjects,
            )

    @abstractmethod
    async def process(self, *args, **kwargs) -> None:
        pass


class AbstractProcessor(ABC):
    def __init__(self) -> None:
        self._service_name = os.getenv("SERVICE_NAME")
        if not self._service_name:
            raise ValueError("SERVICE_NAME environment variable is not set")
        self._cfg = ConfigManager().config_service_name(self._service_name)

    @abstractmethod
    async def process(self, *args, **kwargs) -> None:
        pass
