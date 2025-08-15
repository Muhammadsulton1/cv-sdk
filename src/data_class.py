import os
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class RouterConfig:
    """Конфигурация роутера"""
    nats_hosts: list[str]
    redis_host: str
    redis_port: int
    redis_password: Optional[str] = None
    redis_ssl: bool = False
    topic_input: Optional[str] = None
    service_key: str = "routing_to_models"
    discovery_interval: int = 30
    max_message_size: int = 1024 * 1024  # 1MB
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class InferenceConfing:
    nats_hosts = List[str]
    REDIS_HOST = str
    REDIS_PORT = int
    REGISTER_TTL = int
