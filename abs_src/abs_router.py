import asyncio
import json
import os

from abc import ABC, abstractmethod
from typing import Set, Dict
from src.singeleton.connection_singeleton import NatsClient, RedisClient
from utils.logger import logger


class AbstractRouterManager(ABC):
    def __init__(self):
        """
           Инициализация менеджера маршрутизации.
           Параметры окружения:
               nats_host: Список серверов NATS (по умолчанию: ["nats://localhost:4222"])
               redis_host: Хост Redis (по умолчанию: "localhost")
               redis_port: Порт Redis (по умолчанию: 6379)
               topic_stream: Входной топик для сообщений
        """
        self.nats_host = os.getenv("nats_host", "nats://localhost:4222").split(",")
        self.redis_host = os.getenv('redis_host', 'localhost')
        self.redis_port = int(os.getenv('redis_port', 6379))
        self.service_key = "routing_to_models"

        self.nats_cli = None
        self.redis = None
        self.topic_input = os.getenv('topic_stream')
        self.sub = None
        self.available_models = set()
        self.discovery_interval = 5

    async def __aenter__(self):
        """Асинхронный контекстный менеджер для подключения."""
        self.nats_cli = await NatsClient.connect()
        self.redis = await RedisClient.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Асинхронный контекстный менеджер для отключения."""
        await self.close()
        return None

    async def close(self):
        """Корректно закрывает подключения к NATS и Redis."""
        try:
            if self.nats_cli.is_connected:
                await NatsClient.close()
                self.nats_cli = None
            if self.redis:
                await RedisClient.close()
                self.redis = None
        except Exception as e:
            logger.error(f"Ошибка закрытия соединений: {e}")

    @abstractmethod
    async def _fetch_available_models(self) -> Set[str]:
        """Абстрактный метод для получения списка доступных моделей"""
        pass

    @abstractmethod
    def _prepare_message(self, data: Dict[str, any], model: str) -> Dict[str, any]:
        """Абстрактный метод для подготовки сообщения к публикации"""
        pass

    @abstractmethod
    def _select_models(self, data: Dict[str, any]) -> Set[str]:
        """Абстрактный метод для выбора моделей обработки"""
        pass

    async def _update_available_models(self):
        """
          Периодически обновляет список доступных моделей из Redis.

          Интервал обновления контролируется discovery_interval.
          Сохраняет модели как множество в self.available_models.
        """
        while True:
            try:
                models = await self._fetch_available_models()
                self.available_models = set(models)
                logger.info(f"Обновлен список доступных подписчиков: {self.available_models}")
            except Exception as e:
                logger.error(f"Ошибка обновления доступных подписчиков: {e}")
            await asyncio.sleep(self.discovery_interval)

    async def subscribe(self):
        """Подписывается на входной топик NATS для обработки сообщений."""
        if not self.nats_cli.is_connected:
            return

        self.sub = await self.nats_cli.subscribe(
            subject=f"{self.topic_input}",
            cb=self.message_handler
        )
        logger.info(f"Подписался на топик NATS: {self.topic_input}")

    async def message_handler(self, msg):
        """
          Обработчик входящих сообщений из NATS.

          Параметры:
              msg (NATS.message): Входящее сообщение

          Декодирует данные и передает в publish.
        """
        subject = msg.subject
        data_str = msg.data.decode()
        data = json.loads(data_str)
        logger.info(f"Отправлено сообщение: [{subject}]: {data}")

        await self.publish(data)

    async def publish(self, data):
        """
        Публикует сообщения для всех доступных моделей.

        Параметры:
            data (dict): Данные сообщения

        Формат сообщения:
            frame_id: Идентификатор кадра
            seaweed_url: URL медиаданных
            model: Целевая модель
            timestamp: Временная метка
            cached_key: Ключ кэша метаданных

        Логирует ошибку если нет доступных моделей.
        """
        selected_models = self._select_models(data)
        if not selected_models:
            logger.warning("Нет доступных живых подписчиков для перенаправления кадра")
            return

        for model in selected_models:
            output_topic = f"{model}"
            message = self._prepare_message(data, model)
            await self.nats_cli.publish(output_topic, json.dumps(message).encode())

        logger.info(f"Кадр перенаправлен {data['frame_id']} к {len(selected_models)} моделям")

    @abstractmethod
    async def process(self):
        pass
