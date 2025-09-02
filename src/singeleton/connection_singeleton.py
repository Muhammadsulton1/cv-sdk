import os

import aiohttp
import nats
from nats.aio.errors import ErrConnectionClosed, ErrNoServers, ErrTimeout
from redis.asyncio import Redis

from utils.logger import logger


class NatsClient:
    __instance = None

    @classmethod
    async def connect(cls):
        if cls.__instance is None:
            try:
                cls.__instance = await nats.connect(
                    servers=os.getenv("nats_host", "nats://localhost:4222").split(","),
                    max_reconnect_attempts=-1,
                    reconnect_time_wait=2,
                    connect_timeout=2,
                    ping_interval=5,
                    allow_reconnect=True
                )
                logger.info("Connected to NATS")
            except (ErrNoServers, ErrTimeout, ErrConnectionClosed) as e:
                logger.error(f"Failed to connect to NATS: {e}")
                raise
        return cls.__instance

    @classmethod
    async def close(cls):
        if cls.__instance is not None:
            try:
                await cls.__instance.drain()
                await cls.__instance.close()
                logger.info("NATS connection closed")
            except Exception as e:
                logger.error(f"Error closing NATS: {e}")
            finally:
                cls.__instance = None


class RedisClient:
    __instance = None

    @classmethod
    async def connect(cls):
        if cls.__instance is None:
            try:
                cls.__instance = Redis(
                    host=os.getenv('redis_host', 'localhost'),
                    port=int(os.getenv('redis_port', 6379)),
                    socket_connect_timeout=5,
                    decode_responses=True
                )
                await cls.__instance.ping()
                logger.info("Connected to Redis")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                cls.__instance = None
                raise
        return cls.__instance

    @classmethod
    async def close(cls):
        if cls.__instance is not None:
            try:
                await cls.__instance.aclose()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis: {e}")
            finally:
                cls.__instance = None


class S3Client:
    __instance = None

    @classmethod
    async def connect(cls):
        if cls.__instance is None:
            connector = aiohttp.TCPConnector(
                limit_per_host=100,
                limit=200,
                ttl_dns_cache=300,
                use_dns_cache=True,
                force_close=True,
                enable_cleanup_closed=True
            )

            cls.__instance = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30, connect=10),
                connector=connector,
            )

        return cls.__instance

    @classmethod
    async def close(cls):
        if cls.__instance and not cls.__instance.closed:
            await cls.__instance.close()
