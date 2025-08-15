import asyncio
import time

from abs_src.abs_router import AbstractRouterManager
from utils.logger import logger


class RouterManager(AbstractRouterManager):
    """
    Класс для управления маршрутизацией сообщений между NATS и Redis.
    Наследуется от абстрактного класса AbstractRouterManager

    Основные функции:
    - Подключение к NATS и Redis
    - Динамическое отслеживание доступных моделей
    - Маршрутизация входящих сообщений к моделям
    - Публикация сообщений для обработчиков

    Атрибуты:
        nats_host (list): Список серверов NATS
        redis_host (str): Хост Redis
        redis_port (int): Порт Redis
        service_key (str): Ключ Redis для отслеживания моделей
        nats_cli (NATS): Клиент NATS
        redis (Redis): Асинхронный клиент Redis
        topic_input (str): Входная тема для сообщений
        sub (NATS.subscription): Подписка NATS
        available_models (set): Множество доступных моделей
        discovery_interval (int): Интервал обновления моделей (сек)
    """

    async def _fetch_available_models(self) -> set:
        """Получение моделей из Redis"""
        return await self.redis.smembers(self.service_key)

    def _prepare_message(self, data: dict, model: str) -> dict:
        """Формирование сообщения для модели"""
        return {
            "frame_id": data["frame_id"],
            "seaweed_url": data["seaweed_url"],
            "model": model,
            "timestamp": time.time()
            # "cached_key": f"frame_meta:{data['frame_id']}"
        }

    def _select_models(self, data: dict) -> set:
        """Выбор всех доступных моделей по умолчанию"""
        return self.available_models

    async def run_process(self):
        """Основной цикл (уже внутри контекста)"""
        await self.subscribe()
        await asyncio.create_task(self._update_available_models())
        logger.info("Сервис RoutingManager успешно запущен")
        # while True:
        #     await asyncio.sleep(1)
        await asyncio.Event().wait()

    def process(self):
        async def _run():
            async with self:
                await self.run_process()

        asyncio.run(_run())


if __name__ == '__main__':
    router = RouterManager()
    router.process()
