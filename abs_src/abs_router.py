import asyncio
import json
import os

from abc import ABC, abstractmethod
from typing import Set, Dict
from src.singeleton.connection_singeleton import NatsClient, RedisClient
from src.singeleton.yaml_reader import YamlReader
from utils.decorators import retry_async
from utils.logger import logger


class AbstractRouterManager(ABC):
    """
        Абстрактный базовый класс для роутинга сообщений между NATS и моделями обработки.

        Задачи:
            - Подписываться на входящий топик NATS и принимать сообщения.
            - Динамически обнаруживает доступные модели в Redis.
            - Формирует и публикует сообщения в отдельные топики для каждой модели.

        Основное назначение:
            Служба, обеспечивающая маршрутизацию кадров (frame_id и связанных метаданных)
            к набору сервисов (моделей) для дальнейшей обработки.
        """

    def __init__(self):
        """
        Инициализация настроек и внутренних атрибутов роутера.

        Атрибуты:
            setup_settings (YamlReader): Читатель YAML-конфигурации.
            subscribe_topic (str): Имя входного канала (топика) NATS.
            publish_topic (str): Суффикс исходящего канала NATS.
            nats_cli (NatsClient | None): Клиент для взаимодействия с NATS.
            redis (RedisClient | None): Клиент для взаимодействия с Redis.
            sub (Subscription | None): Объект подписки на топик NATS.
            available_models (Set[str]): Набор доступных моделей для маршрутизации.
            discovery_interval (int): Интервал (с) для обновления списка моделей.
            service_key (str): Ключ в Redis для хранения списка моделей.
        """
        self.setup_settings = YamlReader()
        self.subscribe_topic = self.setup_settings.get(os.getenv('SERVICE_NAME')).get('in_channel')
        self.publish_topic = self.setup_settings.get(os.getenv('SERVICE_NAME')).get('out_channel')

        self.nats_cli = None
        self.redis = None
        self.sub = None
        self.available_models = set()
        self.discovery_interval = 5

        self.service_key = 'models_routing'

    async def __aenter__(self):
        """
        Вход в асинхронный контекст менеджера.

        Открывает соединения с NATS и Redis.

        Возвращает:
            AbstractRouterManager: Экземпляр с установленными соединениями.
        """
        self.nats_cli = await NatsClient.connect()
        self.redis = await RedisClient.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """
        Выход из асинхронного контекста менеджера.

        Закрывает соединения с NATS и Redis.
        """
        await self.close()
        return None

    async def close(self):
        """
        Корректно завершает работу и очищает ресурсы.

        Закрывает клиентские сессии NATS и Redis при их наличии.
        Логирует ошибки, если они возникают в процессе закрытия.
        """
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
        """
        Абстрактный метод для получения актуального списка доступных моделей.

        Ожидается, что реализация будет обращаться к Redis и возвращать множество строк.

        Возвращаемое значение:
            Set[str]: Набор ключей (имен) доступных моделей.
        """
        pass

    @abstractmethod
    def _prepare_message(self, data: Dict[str, any], model: str) -> Dict[str, any]:
        """
        Абстрактный метод для подготовки сообщения к публикации конкретной модели.

        Параметры:
            data (Dict[str, any]): Входные данные из NATS.
            model (str): Имя целевой модели.

        Возвращаемое значение:
            Dict[str, any]: Сообщение, которое будет JSON-сериализовано и отправлено.
        """
        pass

    @abstractmethod
    def _select_models(self, data: Dict[str, any]) -> Set[str]:
        """
        Абстрактный метод для выбора моделей, которым нужно отправить сообщение.

        Логика выбора на основе содержимого data и self.available_models.

        Параметры:
            data (Dict[str, any]): Входное сообщение (декодированное из JSON).

        Возвращаемое значение:
            Set[str]: Множество имен моделей для отправки.
        """
        pass

    async def _update_available_models(self):
        """
        Фоновая задача: периодически обновляет список доступных моделей.

        Запускается единожды при старте сервиса. Каждые discovery_interval секунд:
            - Вызывает _fetch_available_models()
            - Обновляет self.available_models
            - Логирует результат или ошибку
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
        """
        Подписывается на входной топик NATS для получения сообщений.

        Проверяет соединение, создаёт подписку и сохраняет её в self.sub.
        Логирует факт подписки.
        """
        if not self.nats_cli.is_connected:
            return

        self.sub = await self.nats_cli.subscribe(
            subject=f"{self.subscribe_topic}",
            cb=self.message_handler
        )
        logger.info(f"Подписался на топик NATS: {self.subscribe_topic}")

    async def message_handler(self, msg):
        """
            Обработчик сообщений, полученных по подписке.

            Декодирует payload из байт в JSON, логирует и передаёт данные в publish().

            Параметры:
                msg (NATS.message): Объект сообщения от NATS.
        """
        subject = msg.subject
        data_str = msg.data.decode()
        data = json.loads(data_str)
        logger.info(f"Отправлено сообщение: [{subject}]: {data}")

        await self.publish(data)

    @retry_async()
    async def publish(self, data):
        """
        Публикует сообщения для выбранных моделей.

        Логика:
            - Определить список моделей через _select_models()
            - Для каждой модели:
                - Подготовить сообщение через _prepare_message()
                - Опубликовать JSON по топику "{model}_{publish_topic}"

        Параметры:
            data (dict): Входные данные для маршрутизации.

        Особенности:
            - Если нет выбранных моделей, логирует предупреждение и выходит.
            - Повторные попытки в случае временных ошибок.
        """
        selected_models = self._select_models(data)
        if not selected_models:
            logger.warning("Нет доступных живых подписчиков для перенаправления кадра")
            return

        for model in selected_models:
            output_topic = f"{model}_{self.publish_topic}"
            message = self._prepare_message(data, model)
            await self.nats_cli.publish(output_topic, json.dumps(message).encode())

        logger.info(f"Кадр перенаправлен {data['frame_id']} к {len(selected_models)} моделям")

    async def run_process(self):
        """
        Основной цикл работы менеджера.

        Шаги:
            1. Подписаться на входной топик.
            2. Запустить фоновую задачу обновления моделей.
            3. Блокироваться до завершения сервиса.

        Используется внутри асинхронного контекста__aenter__/__aexit__.
        """
        await self.subscribe()
        await asyncio.create_task(self._update_available_models())
        logger.info("Сервис RoutingManager успешно запущен")
        await asyncio.Event().wait()

    def process(self):
        """
        Точка входа при старте сервиса.

        Создаёт и запускает главный асинхронный цикл через asyncio.run().
        """

        async def _run():
            async with self:
                await self.run_process()

        asyncio.run(_run())
