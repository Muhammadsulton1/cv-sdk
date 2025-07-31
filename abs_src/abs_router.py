import asyncio
import json
import os
from abc import ABC, abstractmethod
from redis.asyncio import Redis
from nats.aio.client import Client as NATS
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
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Асинхронный контекстный менеджер для отключения."""
        await self.close()
        return None

    async def connect(self):
        """
        Устанавливает подключения к NATS и Redis.

        Выбрасывает:
            Exception: При ошибках подключения
        """
        self.nats_cli = NATS()
        try:
            await self.nats_cli.connect(
                servers=self.nats_host,
                max_reconnect_attempts=-1,
                reconnect_time_wait=2
            )
            logger.info("Подключения к NATS успешно")
        except Exception as e:
            logger.error(f"Подключения к NATS упал с ошибкой: {e}")
            raise

        try:
            self.redis = Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True
            )
            await self.redis.ping()
            logger.info("Подключения к Redis успешно")
        except Exception as e:
            logger.error(f"Подключения к Redis упал с ошибкой: {e}")
            raise

    async def close(self):
        """Корректно закрывает подключения к NATS и Redis."""
        try:
            if self.nats_cli.is_connected:
                await self.nats_cli.drain()
            if self.redis:
                await self.redis.aclose()
        except Exception as e:
            logger.error(f"Ошибка закрытия соединений: {e}")
        finally:
            await self.nats_cli.close()

    @abstractmethod
    async def _fetch_available_models(self) -> set:
        """Абстрактный метод для получения списка доступных моделей"""
        pass

    @abstractmethod
    def _prepare_message(self, data: dict, model: str) -> dict:
        """Абстрактный метод для подготовки сообщения к публикации"""
        pass

    @abstractmethod
    def _select_models(self, data: dict) -> set:
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
