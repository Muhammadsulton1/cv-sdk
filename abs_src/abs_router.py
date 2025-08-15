import asyncio
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Set, Dict, Any, Optional

from redis.asyncio import Redis
from redis.asyncio import ConnectionError as RedisConnectionError
from redis import RedisError
from nats.aio.client import Client as NATS
from nats.aio.errors import ErrConnectionClosed, ErrNoServers, ErrTimeout, ErrBadSubscription

from src.singeleton.yaml_reader import YamlReader
from utils.decorators import measure_latency_async
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
        self.redis_password = os.getenv('redis_password', '12345abc!')
        self.service_key = "routing_to_models"

        self.nats_cli = None
        self.redis = None
        self.sub = None
        self.available_models = set()
        self.discovery_interval = 5

        self.setup_config = YamlReader()
        self.in_channel = self.setup_config.get('DataRouter')['in_channel']
        self.out_channel = self.setup_config.get('DataRouter')['out_channel']

    async def __aenter__(self):
        """Асинхронный контекстный менеджер для подключения."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Асинхронный контекстный менеджер для отключения."""
        await self.close()
        return None

    async def connect_nats(self) -> None:
        """
            Устанавливает подключение к Redis
            Выбрасывает:
                Exception: При ошибках подключения
        """
        self.nats_cli = NATS()
        try:
            await self.nats_cli.connect(
                servers=self.nats_host,
                max_reconnect_attempts=-1,
                reconnect_time_wait=10,
            )
        except (ErrConnectionClosed, ErrNoServers, ErrTimeout) as err:
            logger.error(f"NATS не удалось подключиться: {err}")
            self.nats_cli = None
            raise

    async def connect_redis(self) -> None:
        """
        Устанавливает подключение к Redis
        Выбрасывает:
            Exception: При ошибках подключения
        """
        try:
            self.redis = Redis(host=self.redis_host,
                               port=self.redis_port,
                               socket_connect_timeout=5,
                               decode_responses=True)
        except (RedisError, RedisConnectionError) as err:
            logger.error(f"Redis не удалось подключиться: {err}")
            self.redis = None
            raise

    async def connect(self):
        """
         Устанавливает подключения к NATS и Redis.

         Выбрасывает:
             Exception: При ошибках подключения
         """
        await self.connect_nats()
        await self.connect_redis()

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
            self.redis = None
            self.nats_cli = None

    @abstractmethod
    async def _fetch_available_models(self) -> Set[str]:
        """Абстрактный метод для получения списка доступных моделей"""
        pass

    @abstractmethod
    def _prepare_message(self, data: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Абстрактный метод для подготовки сообщения к публикации"""
        pass

    @abstractmethod
    def _select_models(self, data: Dict[str, Any]) -> Set[str]:
        """Абстрактный метод для выбора моделей обработки"""
        pass

    @measure_latency_async()
    async def _update_available_models(self) -> None:
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

    @measure_latency_async()
    async def subscribe(self) -> None:
        """Подписывается на входной топик NATS для обработки сообщений."""
        if not self.nats_cli.is_connected:
            return

        try:
            self.sub = await self.nats_cli.subscribe(
                subject=f"{self.in_channel}",
                cb=self.message_handler
            )
            logger.info(f"Подписался на топик NATS: {self.in_channel}")
        except ErrBadSubscription as err:
            logger.error("Не смог подписаться на заданный топик", str(err))
            raise

    @measure_latency_async()
    async def message_handler(self, msg) -> None:
        """
          Обработчик входящих сообщений из NATS.

          Параметры:
              msg (NATS.message): Входящее сообщение

          Декодирует данные и передает в publish.
        """
        subject = msg.subject
        data_str = msg.data.decode()
        data = json.loads(data_str)
        logger.debug(f"Отправлено сообщение: [{subject}]: {data}")

        await self.publish(data)

    @measure_latency_async()
    async def publish(self, data: Dict[str, Any]) -> None:
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
            output_topic = f"{model}_{self.out_channel}"
            message = self._prepare_message(data, model)
            await self.nats_cli.publish(output_topic, json.dumps(message).encode())

        logger.debug(f"Кадр перенаправлен {data['frame_id']} на инференс {selected_models} моделям")

    @abstractmethod
    async def process(self):
        pass
