import asyncio
import json
from nats.aio.client import Client as NATS
from nats.aio.errors import ErrConnectionClosed, ErrTimeout
import redis
import time
from utils.logger import logger


class Router:
    def __init__(self,
                 nats_servers=["nats://localhost:4222"],
                 redis_host="localhost",
                 redis_port=6379,
                 input_topic="frames-stream",
                 output_topic_prefix="inference",
                 max_concurrent=50,
                 cache_ttl=600):
        """
        Инициализация роутера

        :param nats_servers: Список серверов NATS
        :param redis_host: Хост Redis для кэширования
        :param redis_port: Порт Redis
        :param input_topic: Входной топик для кадров
        :param output_topic_prefix: Префикс выходных топиков
        :param max_concurrent: Максимальное количество одновременных задач
        :param cache_ttl: TTL кэша в секундах
        """
        self.nats_servers = nats_servers
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.input_topic = input_topic
        self.output_topic_prefix = output_topic_prefix
        self.max_concurrent = max_concurrent
        self.cache_ttl = cache_ttl

        self.nc = NATS()
        self.redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=False,
            max_connections=20
        )
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.logger = logger
        self.processed_cache = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=1,  # Отдельная БД для дедупликации
            decode_responses=True
        )

    async def connect(self):
        """Подключение к NATS и Redis"""
        try:
            # Подключаемся к NATS
            await self.nc.connect(
                servers=self.nats_servers,
                max_reconnect_attempts=-1,  # Бесконечные попытки
                reconnect_time_wait=2,
            )
            self.logger.info(f"Connected to NATS at {self.nats_servers}")

            # Проверяем подключение к Redis
            if not self.redis.ping():
                raise ConnectionError("Redis connection failed")
            self.logger.info(f"Connected to Redis at {self.redis_host}:{self.redis_port}")

        except Exception as e:
            self.logger.error(f"Connection error: {str(e)}")
            raise

    async def message_handler(self, msg):
        """Обработчик входящих сообщений"""
        try:
            async with self.semaphore:  # Ограничение параллелизма
                data = json.loads(msg.data.decode())
                self.logger.debug(f"Received message: {data['frame_id']}")

                # Дедупликация сообщений
                if self.is_duplicate(data["frame_id"]):
                    self.logger.debug(f"Duplicate frame skipped: {data['frame_id']}")
                    return

                # Кэширование метаданных
                self.cache_frame_metadata(data)

                # Маршрутизация для каждой модели
                await self.route_to_models(data)

        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")

    def is_duplicate(self, frame_id):
        """Проверка дубликатов через Redis"""
        key = f"frame:{frame_id}"
        if self.processed_cache.exists(key):
            return True

        # Устанавливаем ключ с TTL
        self.processed_cache.setex(key, self.cache_ttl, "1")
        return False

    def cache_frame_metadata(self, data):
        """Кэширование метаданных кадра в Redis"""
        try:
            key = f"frame_meta:{data['frame_id']}"
            value = {
                "seaweed_url": data["seaweed_url"],
                "timestamp": data["timestamp"],
                "models": ",".join(data["models"])
            }
            self.redis.hset(key, mapping=value)
            self.redis.expire(key, self.cache_ttl)
        except Exception as e:
            self.logger.error(f"Caching failed: {str(e)}")

    async def route_to_models(self, data):
        """Перенаправление кадра в топики для моделей"""
        tasks = []
        for model in data["models"]:
            output_topic = f"{self.output_topic_prefix}.{model}"

            # Формируем оптимизированное сообщение
            message = {
                "frame_id": data["frame_id"],
                "seaweed_url": data["seaweed_url"],
                "model": model,
                "timestamp": time.time(),
                "cached_key": f"frame_meta:{data['frame_id']}"  # Для быстрого доступа
            }

            # Асинхронная отправка
            task = asyncio.create_task(
                self.nc.publish(output_topic, json.dumps(message).encode())
            )
            tasks.append(task)

        # Ожидаем завершения всех отправок
        await asyncio.gather(*tasks)
        self.logger.info(f"Routed frame {data['frame_id']} to {len(data['models'])} models")

    async def subscribe(self):
        """Подписка на входной топик"""
        await self.nc.subscribe(
            self.input_topic,
            cb=self.message_handler,
            config={
                "deliver_policy": "all",  # Получать все сообщения
                "ack_policy": "explicit",  # Ручное подтверждение
                "max_ack_pending": 1000,  # Макс. неподтвержденных сообщений
            }
        )
        self.logger.info(f"Subscribed to {self.input_topic}")

    async def run(self):
        """Основной цикл работы роутера"""
        await self.connect()
        await self.subscribe()
        self.logger.info("Router started. Press Ctrl+C to exit.")

        # Бесконечный цикл обработки
        while True:
            await asyncio.sleep(1)

    async def close(self):
        """Корректное завершение работы"""
        await self.nc.drain()
        self.redis.close()
        self.processed_cache.close()
        self.logger.info("Router shutdown complete")


if __name__ == "__main__":
    router = Router(
        nats_servers=["nats://localhost:4222"],
        redis_host="localhost",
        input_topic="frames-stream",
        output_topic_prefix="inference",
        max_concurrent=100
    )

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(router.run())
    except KeyboardInterrupt:
        loop.run_until_complete(router.close())
    finally:
        loop.close()

if __name__ == "__main__":
    router = Router(
        nats_servers=["nats://localhost:4222"],
        redis_host="localhost",
        input_topic="frames-stream",
        output_topic_prefix="inference",
        max_concurrent=100
    )

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(router.run())
    except KeyboardInterrupt:
        loop.run_until_complete(router.close())
    finally:
        loop.close()