import asyncio
import os
from typing import Optional
from urllib.parse import urlparse

import aioboto3
import nats
from botocore.config import Config
from nats.aio.errors import ErrConnectionClosed, ErrNoServers, ErrTimeout
from redis.asyncio import Redis

from common.utils.err import NatsConnectionError
from common.utils.logger import logger


class NatsClient:
    __instance = None
    __lock = asyncio.Lock()

    @classmethod
    async def connect(cls):
        if cls.__instance is not None and not cls.__instance.is_closed:
            return cls.__instance

        async with cls.__lock:
            if cls.__instance is not None and not cls.__instance.is_closed:
                return cls.__instance

            cls._validate()
            servers = cls._build_servers()

            try:
                cls.__instance = await nats.connect(
                    servers=servers,
                    max_reconnect_attempts=-1,
                    reconnect_time_wait=2,
                    connect_timeout=2,
                    ping_interval=5,
                    allow_reconnect=True,
                )
                logger.info("Connected to NATS successfully")
                return cls.__instance

            except (ErrNoServers, ErrTimeout, ErrConnectionClosed) as e:
                logger.exception("Failed to connect to NATS")
                raise NatsConnectionError("Failed to connect to NATS") from e

    @classmethod
    async def get_connection(cls):
        return await cls.connect()

    @classmethod
    async def get_jetstream(cls, **kwargs):
        nc = await cls.connect()
        return nc.jetstream(**kwargs)

    @classmethod
    async def get_jsm(cls, **kwargs):
        nc = await cls.connect()
        return nc.jsm(**kwargs)

    @classmethod
    def _validate(cls) -> None:
        required = ("NATS_HOST", "NATS_PORT")
        missing = [k for k in required if not os.getenv(k)]
        if missing:
            raise ValueError(f"Missing NATS environment variables: {', '.join(missing)}")

    @classmethod
    def _build_servers(cls) -> list[str]:
        hosts = [h.strip() for h in os.getenv("NATS_HOST", "").split(",") if h.strip()]
        port = os.getenv("NATS_PORT")
        return [f"nats://{host}:{port}" for host in hosts]

    @classmethod
    async def close(cls):
        if cls.__instance is not None:
            try:
                await cls.__instance.drain()
                logger.info("NATS connection drained and closed")
            except Exception:
                logger.exception("Error closing NATS")
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
    """
    Синглтон-клиент для aioboto3.
    Ref-counted: первый connect() поднимает соединение,
    последний close() его закрывает.
    """
    __instance = None
    _client_context = None
    _refcount: int = 0
    _connect_lock: Optional[asyncio.Lock] = None

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        if cls._connect_lock is None:
            cls._connect_lock = asyncio.Lock()
        return cls._connect_lock

    @classmethod
    def _endpoint_url(cls) -> str:
        host = os.getenv("S3_HOST", "localhost")
        port = os.getenv("S3_PORT", "9000")
        if host.startswith(("http://", "https://")):
            parsed = urlparse(host)
            return host.rstrip("/") if parsed.port else f"{host.rstrip('/')}:{port}"
        return f"http://{host}:{port}"

    @classmethod
    def _validate_credentials(cls) -> None:
        missing = [k for k in ("S3_ACCESS_KEY", "S3_SECRET_KEY", "S3_HOST") if not os.getenv(k)]
        if missing:
            raise ValueError(f"Missing S3 env vars: {', '.join(missing)}")

    @classmethod
    async def connect(cls) -> "S3Client":
        async with cls._get_lock():
            if cls.__instance is None:
                cls._validate_credentials()
                cfg = Config(
                    connect_timeout=int(os.getenv("S3_CONNECT_TIMEOUT", "20")),
                    read_timeout=int(os.getenv("S3_READ_TIMEOUT", "20")),
                    max_pool_connections=int(os.getenv("S3_MAX_POOL_CONNECTIONS", "32")),
                )
                session = aioboto3.Session()
                cls._client_context = session.client(
                    service_name="s3",
                    endpoint_url=cls._endpoint_url(),
                    aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
                    aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
                    region_name=os.getenv("AWS_DEFAULT_REGION", "eu-west-1"),
                    config=cfg,
                )
                try:
                    cls.__instance = await cls._client_context.__aenter__()
                    logger.info("S3 client connected")
                except Exception:
                    cls._client_context = None
                    cls.__instance = None
                    raise
            cls._refcount += 1
        return cls.__instance

    @classmethod
    async def close(cls) -> None:
        async with cls._get_lock():
            if cls._refcount <= 0:
                return
            cls._refcount -= 1
            if cls._refcount > 0:
                return
            if cls._client_context is not None:
                try:
                    await cls._client_context.__aexit__(None, None, None)
                    logger.info("S3 client closed")
                except Exception as e:
                    logger.error(f"Error closing S3 client: {e}")
                finally:
                    cls.__instance = None
                    cls._client_context = None


if __name__ == "__main__":
    async def main():
        client = await NatsClient.connect()
    asyncio.run(main())