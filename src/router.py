import time
from typing import Dict, Any

from abs_src.abs_router import AbstractRouterManager


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

    def _prepare_message(self, data: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Формирование сообщения для модели"""
        return {
            "frame_id": data["frame_id"],
            "seaweed_url": data["seaweed_url"],
            "model": model,
            "timestamp": time.time()
        }

    def _select_models(self, data: Dict[str, Any]) -> set:
        """Выбор всех доступных моделей по умолчанию"""
        return self.available_models


if __name__ == '__main__':
    router = RouterManager()
    router.process()
